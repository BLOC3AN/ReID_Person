#!/usr/bin/env python3
"""
Detection Service API
FastAPI service for person detection, tracking, and re-identification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Add scripts directory to path
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(scripts_dir))

import os
import shutil
import csv
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
from loguru import logger
import threading
import time
import asyncio
import json

# Import modules
from scripts.detect_and_track import PersonReIDPipeline
from scripts.zone_monitor import process_video_with_zones, process_multi_stream_with_zones
from core.preloaded_manager import preloaded_manager
from utils.multi_stream_reader import parse_stream_urls

app = FastAPI(title="Detection Service", version="1.0.0")

# Storage paths (use relative paths for development, absolute for Docker)
project_root = Path(__file__).parent.parent
UPLOAD_DIR = project_root / "data" / "uploads"
OUTPUT_DIR = project_root / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job storage
jobs = {}
# Progress tracking
# Structure for single-stream: {"current_frame": int, "total_frames": int, "tracks": [...]}
# Structure for multi-stream: {"cameras": {0: {"current_frame": int, "tracks": [...]}, 1: {...}}}
progress_data = {}
# Cancellation flags
cancellation_flags = {}  # {job_id: threading.Event()}
# WebSocket connections for real-time violation logs
websocket_connections: Dict[str, List[WebSocket]] = {}  # {job_id: [websocket1, websocket2, ...]}


# Helper functions
def _save_zone_config(zone_config: UploadFile, job_id: str) -> Optional[Path]:
    """
    Save uploaded zone config file

    Args:
        zone_config: Uploaded zone config file
        job_id: Job ID for filename

    Returns:
        Path to saved zone config file or None if no config provided
    """
    if zone_config is None:
        return None

    # Preserve original file extension (.yaml, .yml, or .json)
    original_ext = Path(zone_config.filename).suffix
    if original_ext not in ['.yaml', '.yml', '.json']:
        original_ext = '.yaml'  # Default to .yaml if unknown
    zone_config_filename = f"{job_id}_zones{original_ext}"
    zone_config_path = UPLOAD_DIR / zone_config_filename

    with open(zone_config_path, "wb") as buffer:
        shutil.copyfileobj(zone_config.file, buffer)

    logger.info(f"Zone config saved: {zone_config_path}")
    return zone_config_path


def _ensure_output_directories():
    """Ensure all output directories exist"""
    (OUTPUT_DIR / "videos").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)


def _get_total_frames(video_path: str) -> int:
    """
    Get total frames from video file

    Args:
        video_path: Path to video file

    Returns:
        Total frame count (0 for streams)
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


@app.on_event("startup")
async def startup_event():
    """Initialize pre-loaded components at service startup"""
    logger.info("🚀 Detection Service starting up...")
    try:
        # Ensure output directories exist (important for Docker volume mounts)
        _ensure_output_directories()
        logger.info(f"✅ Output directories ready: {OUTPUT_DIR}")

        # Initialize pre-loaded components
        preloaded_manager.initialize()
        logger.info("✅ Detection Service ready with pre-loaded components")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pre-loaded components: {e}")
        logger.warning("⚠️  Service will fall back to lazy loading")


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed, cancelled
    message: Optional[str] = None
    output_video: Optional[str] = None
    output_csv: Optional[str] = None
    output_log: Optional[str] = None
    output_json: Optional[str] = None  # Zone monitoring JSON report
    zone_monitoring: bool = False
    error: Optional[str] = None


class ProgressStatus(BaseModel):
    job_id: str
    status: str
    current_frame: int = 0
    total_frames: int = 0
    progress_percent: float = 0.0
    tracks: List[dict] = []
    violations: List[dict] = []  # Real-time violations
    message: Optional[str] = None
    cameras: Optional[Dict] = None  # Per-camera progress for multi-stream


def process_detection(job_id: str, video_path: str, output_video: str,
                      output_csv: str, output_log: str, config_path: Optional[str],
                      similarity_threshold: float = 0.8, model_type: Optional[str] = None,
                      conf_thresh: Optional[float] = None, track_thresh: Optional[float] = None,
                      face_conf_thresh: Optional[float] = None,
                      zone_config_path: Optional[str] = None, iou_threshold: float = 0.6,
                      zone_opacity: float = 0.3, max_frames: Optional[int] = None,
                      max_duration_seconds: Optional[int] = None, alert_threshold: float = 0,
                      zone_workers: Optional[int] = None):
    """
    Background task to process detection and tracking with optional zone monitoring
    Auto-detects single vs multi-stream and processes accordingly
    """
    # Initialize output_json to None (will be set if zone monitoring is used)
    output_json = None

    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting detection job {job_id}")

        # Initialize cancellation flag
        cancellation_flags[job_id] = threading.Event()

        # Parse video_path to detect single vs multi-stream
        stream_urls = parse_stream_urls(video_path)
        num_streams = len(stream_urls)
        is_multi_stream = num_streams > 1

        logger.info(f"Detected {num_streams} stream(s): {stream_urls}")

        # Initialize progress tracking
        if is_multi_stream:
            # Multi-stream: per-camera progress
            progress_data[job_id] = {
                "cameras": {},
                "violations": [],
                "last_update": time.time(),
                "num_streams": num_streams
            }
            for i in range(num_streams):
                progress_data[job_id]["cameras"][i] = {
                    "current_frame": 0,
                    "tracks": []
                }
        else:
            # Single-stream: backward compatible
            progress_data[job_id] = {
                "current_frame": 0,
                "total_frames": 0,
                "tracks": [],
                "violations": [],
                "last_update": time.time()
            }

        # Check if zone monitoring is enabled
        use_zones = zone_config_path is not None and Path(zone_config_path).exists()

        if use_zones:
            logger.info(f"Zone monitoring enabled: {zone_config_path}")

            # Check if multi-stream + zone monitoring
            if is_multi_stream:
                logger.info(f"🎯 Multi-stream zone monitoring: {num_streams} cameras")

                # Create multi-stream output directory structure with timestamp (UTC+7)
                from datetime import datetime, timezone, timedelta
                tz_hcm = timezone(timedelta(hours=7))
                timestamp = datetime.now(tz_hcm).strftime("%Y-%m-%d-%H-%M")
                multi_stream_dirname = f"multi_stream_{timestamp}"
                multi_stream_dir = OUTPUT_DIR / multi_stream_dirname
                multi_stream_dir.mkdir(parents=True, exist_ok=True)

                # Store directory name in job for later retrieval
                jobs[job_id]["multi_stream_dir"] = multi_stream_dirname

                # Multi-stream zone monitoring: parallel processing per camera
                results = process_multi_stream_with_zones(
                    stream_urls=stream_urls,
                    zone_config_path=zone_config_path,
                    reid_config_path=config_path,
                    similarity_threshold=similarity_threshold,
                    iou_threshold=iou_threshold,
                    zone_opacity=zone_opacity,
                    output_dir=str(multi_stream_dir),
                    max_frames=max_frames,
                    max_duration_seconds=max_duration_seconds,
                    progress_callback=lambda cam_id, frame_id, tracks: _update_progress(job_id, frame_id, tracks, camera_id=cam_id),
                    violation_callback=lambda violation: _add_violation(job_id, violation),
                    cancellation_flag=cancellation_flags.get(job_id),
                    alert_threshold=alert_threshold,
                    zone_workers=zone_workers
                )

                # Create summary log file
                with open(output_log, 'w') as f:
                    f.write(f"Multi-Stream Zone Monitoring Detection Log\n")
                    f.write(f"Job ID: {job_id}\n")
                    f.write(f"Number of Cameras: {num_streams}\n")
                    f.write(f"Zone Config: {zone_config_path}\n")
                    f.write(f"IoP Threshold: {iou_threshold}\n")
                    f.write(f"Similarity Threshold: {similarity_threshold}\n")
                    f.write(f"\nOutput Directory: {multi_stream_dir}\n")
                    f.write(f"Aggregated CSV: {output_csv}\n")
                    f.write(f"Aggregated JSON: {output_json}\n")
                    f.write(f"Main Video: {output_video}\n")
                    f.write(f"\nPer-Camera Results:\n")
                    for cam_id, result in results['cameras'].items():
                        f.write(f"\n  Camera {cam_id}:\n")
                        f.write(f"    Status: {result['status']}\n")
                        if result['status'] == 'completed':
                            f.write(f"    Video: {result['output_video']}\n")
                            f.write(f"    CSV: {result['output_csv']}\n")
                            f.write(f"    JSON: {result['output_json']}\n")
                        else:
                            f.write(f"    Error: {result.get('error', 'Unknown')}\n")
                    f.write(f"\nTotal Time: {results['total_time']:.1f}s\n")
                    f.write(f"Status: {results['status']}\n")

                # Create aggregated CSV file (combine all cameras)
                with open(output_csv, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    # Use the same header as zone_monitor.py
                    writer.writerow([
                        'frame_id', 'track_id', 'global_id', 'person_name', 'similarity',
                        'x', 'y', 'w', 'h', 'zone_id', 'zone_name', 'duration_in_zone', 'camera_idx'
                    ])

                    for cam_id, result in results['cameras'].items():
                        if result['status'] == 'completed' and os.path.exists(result['output_csv']):
                            with open(result['output_csv'], 'r') as cam_csv:
                                reader = csv.DictReader(cam_csv)
                                for row in reader:
                                    # Copy all fields, update camera_idx to reflect actual camera ID
                                    writer.writerow([
                                        row.get('frame_id', ''),
                                        row.get('track_id', ''),
                                        row.get('global_id', ''),
                                        row.get('person_name', ''),
                                        row.get('similarity', ''),
                                        row.get('x', ''),
                                        row.get('y', ''),
                                        row.get('w', ''),
                                        row.get('h', ''),
                                        row.get('zone_id', ''),
                                        row.get('zone_name', ''),
                                        row.get('duration_in_zone', ''),
                                        cam_id  # Use actual camera ID from multi-stream
                                    ])

                # Create aggregated JSON file (combine all cameras)
                output_json = str(Path(output_log).parent / f"{job_id}_zones.json")
                aggregated_json = {
                    'job_id': job_id,
                    'num_cameras': num_streams,
                    'total_time': results['total_time'],
                    'status': results['status'],
                    'cameras': {}
                }

                for cam_id, result in results['cameras'].items():
                    if result['status'] == 'completed' and os.path.exists(result['output_json']):
                        with open(result['output_json'], 'r') as cam_json:
                            aggregated_json['cameras'][f'camera_{cam_id}'] = json.load(cam_json)
                    else:
                        aggregated_json['cameras'][f'camera_{cam_id}'] = {
                            'status': result['status'],
                            'error': result.get('error', 'Unknown')
                        }

                with open(output_json, 'w') as f:
                    json.dump(aggregated_json, f, indent=2)

                # For video output: copy the first successful camera's video
                # (Merging videos would be too complex and time-consuming)
                video_copied = False
                for cam_id in sorted(results['cameras'].keys()):
                    result = results['cameras'][cam_id]
                    if result['status'] == 'completed' and os.path.exists(result['output_video']):
                        # Copy first successful camera video as the main output
                        shutil.copy2(result['output_video'], output_video)
                        logger.info(f"📹 Copied camera {cam_id} video as main output: {output_video}")
                        video_copied = True
                        break

                if not video_copied:
                    # If no video was copied, create a text file with info
                    with open(output_video.replace('.mp4', '.txt'), 'w') as f:
                        f.write(f"Multi-Stream Zone Monitoring - No video available\n")
                        f.write(f"Job ID: {job_id}\n")
                        f.write(f"All cameras failed to produce video output.\n")

            else:
                # Single-stream zone monitoring
                logger.info("Single-stream zone monitoring")

                # Get total frames
                total_frames = _get_total_frames(video_path)
                progress_data[job_id]["total_frames"] = total_frames

                # Output JSON for zone report
                output_json = str(Path(output_log).parent / f"{job_id}_zones.json")

                # Run with zone monitoring
                process_video_with_zones(
                    video_path=video_path,
                    zone_config_path=zone_config_path,
                    reid_config_path=config_path,
                    similarity_threshold=similarity_threshold,
                    iou_threshold=iou_threshold,
                    zone_opacity=zone_opacity,
                    output_video_path=output_video,
                    output_csv_path=output_csv,
                    output_json_path=output_json,
                    max_frames=max_frames,
                    max_duration_seconds=max_duration_seconds,
                    progress_callback=lambda frame_id, tracks: _update_progress(job_id, frame_id, tracks),
                    violation_callback=lambda violation: _add_violation(job_id, violation),
                    cancellation_flag=cancellation_flags.get(job_id),
                    alert_threshold=alert_threshold,
                    zone_workers=zone_workers
                )

                # Create a simple log file for zone monitoring
                # (zone_monitor uses loguru which outputs to console, not file)
                with open(output_log, 'w') as f:
                    f.write(f"Zone Monitoring Detection Log\n")
                    f.write(f"Job ID: {job_id}\n")
                    f.write(f"Video: {video_path}\n")
                    f.write(f"Zone Config: {zone_config_path}\n")
                    f.write(f"IoP Threshold: {iou_threshold}\n")
                    f.write(f"Similarity Threshold: {similarity_threshold}\n")
                    f.write(f"\nOutputs:\n")
                    f.write(f"  Video: {output_video}\n")
                    f.write(f"  CSV: {output_csv}\n")
                    f.write(f"  JSON Report: {output_json}\n")
                    f.write(f"\nStatus: Completed\n")
                    f.write(f"\nNote: Detailed logs are in the JSON report.\n")

        else:
            # Standard detection without zones
            logger.info("Standard detection (no zone monitoring)")

            # Initialize pipeline (automatically uses pre-loaded components if available)
            pipeline = PersonReIDPipeline(config_path=config_path)
            output_json = None  # No JSON output for standard detection

            # Override config parameters if provided (for this job only)
            if model_type is not None:
                pipeline.config['detection']['model_type'] = model_type

            if conf_thresh is not None:
                pipeline.config['detection']['conf_threshold'] = conf_thresh

            if track_thresh is not None:
                pipeline.config['tracking']['track_thresh'] = track_thresh

            if face_conf_thresh is not None:
                pipeline.config['reid']['triton']['face_conf_threshold'] = face_conf_thresh
                # Update SCRFD client threshold directly (for pre-loaded components)
                if hasattr(pipeline.extractor, 'face_detector'):
                    pipeline.extractor.face_detector.conf_threshold = face_conf_thresh
                    logger.info(f"✅ Updated face confidence threshold to {face_conf_thresh}")

            # Components are either pre-loaded or will be initialized automatically

            # Multi-stream without zones is not supported (always use zone monitoring for multi-stream)
            if is_multi_stream:
                logger.error(f"❌ Multi-stream detection requires zone monitoring enabled")
                jobs[job_id]["status"] = "failed"
                jobs[job_id]["message"] = "Multi-stream detection requires zone monitoring. Please enable zone monitoring or use single stream."
                return

            # Single-stream processing
            logger.info("📹 Single-stream mode")

            # Get total frames from video (will be 0 for streams)
            total_frames = _get_total_frames(video_path)
            if not is_multi_stream:
                progress_data[job_id]["total_frames"] = total_frames

            # Run detection and tracking with specific output paths
            pipeline.process_video(
                video_path=video_path,
                similarity_threshold=similarity_threshold,
                output_video_path=output_video,
                output_csv_path=output_csv,
                output_log_path=output_log,
                max_frames=max_frames,
                max_duration_seconds=max_duration_seconds,
                progress_callback=lambda frame_id, tracks: _update_progress(job_id, frame_id, tracks),
                cancellation_flag=cancellation_flags.get(job_id)
            )

        # Check if job was cancelled
        if job_id in cancellation_flags and cancellation_flags[job_id].is_set():
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["message"] = "Job cancelled by user"
            logger.info(f"Detection job {job_id} cancelled")
        else:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["message"] = "Detection and tracking completed successfully"
            logger.info(f"Detection job {job_id} completed")

        # Update job outputs (for both completed and cancelled jobs)
        jobs[job_id]["output_video"] = output_video
        jobs[job_id]["output_csv"] = output_csv
        jobs[job_id]["output_log"] = output_log
        if output_json:
            jobs[job_id]["output_json"] = output_json

        # Mark if this is a multi-stream job (for UI to know whether to show ZIP download)
        jobs[job_id]["is_multi_stream"] = is_multi_stream

    except Exception as e:
        # Check if exception was due to cancellation
        if job_id in cancellation_flags and cancellation_flags[job_id].is_set():
            jobs[job_id]["status"] = "cancelled"
            jobs[job_id]["message"] = "Job cancelled by user"
            logger.info(f"Detection job {job_id} cancelled (exception during cancellation)")
        else:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = str(e)
            logger.error(f"Detection job {job_id} failed: {e}")
    finally:
        # Cleanup
        if job_id in progress_data:
            progress_data[job_id]["cleanup_time"] = time.time()
        if job_id in cancellation_flags:
            del cancellation_flags[job_id]


def _update_progress(job_id: str, frame_id: int, tracks: List[dict], camera_id: Optional[int] = None):
    """
    Update progress data for a job (supports both single and multi-camera)

    Args:
        job_id: Job ID
        frame_id: Current frame ID
        tracks: List of track dictionaries
        camera_id: Camera ID for multi-camera (None for single camera)
    """
    if job_id in progress_data:
        if camera_id is not None:
            # Multi-camera mode
            if "cameras" not in progress_data[job_id]:
                progress_data[job_id]["cameras"] = {}

            if camera_id not in progress_data[job_id]["cameras"]:
                progress_data[job_id]["cameras"][camera_id] = {}

            progress_data[job_id]["cameras"][camera_id]["current_frame"] = frame_id
            progress_data[job_id]["cameras"][camera_id]["tracks"] = tracks
            progress_data[job_id]["cameras"][camera_id]["last_update"] = time.time()
        else:
            # Single-camera mode (backward compatible)
            progress_data[job_id]["current_frame"] = frame_id
            progress_data[job_id]["tracks"] = tracks
            progress_data[job_id]["last_update"] = time.time()


def _add_violation(job_id: str, violation: dict):
    """Add violation to progress data for real-time alerts (ZONE-CENTRIC LOGIC)"""
    if job_id in progress_data:
        progress_data[job_id]["violations"].append(violation)

        # Format violation message based on type
        if violation.get('type') == 'zone_incomplete':
            # Zone-centric violation
            missing_str = ", ".join([f"{name} (ID:{pid})"
                                    for pid, name in zip(violation['missing_persons'], violation['missing_names'])])
            logger.warning(f"🚨 [Job {job_id}] ZONE VIOLATION: Zone '{violation['zone_name']}' "
                          f"incomplete - Missing: {missing_str} at frame {violation['frame_id']}")
        else:
            # Legacy person-centric violation (backward compatibility)
            logger.warning(f"🚨 [Job {job_id}] VIOLATION: {violation.get('person_name', 'Unknown')} "
                          f"entered unauthorized zone '{violation['zone_name']}' at frame {violation['frame_id']}")

        # Broadcast to WebSocket clients (schedule in event loop)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(_broadcast_violation(job_id, violation), loop)
        except Exception as e:
            logger.debug(f"Could not broadcast violation to WebSocket: {e}")


async def _broadcast_violation(job_id: str, violation: dict):
    """Broadcast violation to all connected WebSocket clients for this job"""
    if job_id in websocket_connections:
        # Format log message
        timestamp = time.strftime("%H:%M:%S")

        if violation.get('type') == 'zone_incomplete':
            missing_str = ", ".join(violation['missing_names'])
            log_msg = {
                "timestamp": timestamp,
                "level": "error",
                "zone": violation['zone_name'],
                "message": f"Zone incomplete: Missing {missing_str}",
                "frame": violation['frame_id']
            }
        else:
            log_msg = {
                "timestamp": timestamp,
                "level": "error",
                "zone": violation['zone_name'],
                "message": f"{violation.get('person_name', 'Unknown')} entered unauthorized zone",
                "frame": violation['frame_id']
            }

        # Send to all connected clients
        disconnected = []
        for ws in websocket_connections[job_id]:
            try:
                await ws.send_json(log_msg)
            except Exception as e:
                logger.debug(f"Failed to send to WebSocket: {e}")
                disconnected.append(ws)

        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections[job_id].remove(ws)


@app.get("/")
async def root():
    return {"service": "Detection Service", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.websocket("/ws/violations/{job_id}")
async def websocket_violations(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint for real-time violation logs

    Clients connect to this endpoint to receive violation events as they occur.
    Messages are sent in JSON format:
    {
        "timestamp": "HH:MM:SS",
        "level": "error" | "info",
        "zone": "Zone_A",
        "message": "Zone incomplete: Missing Khiem",
        "frame": 123
    }
    """
    await websocket.accept()

    # Add to connections list
    if job_id not in websocket_connections:
        websocket_connections[job_id] = []
    websocket_connections[job_id].append(websocket)

    logger.info(f"WebSocket client connected for job {job_id}")

    try:
        # Send initial connection message
        await websocket.send_json({
            "timestamp": time.strftime("%H:%M:%S"),
            "level": "info",
            "zone": "System",
            "message": f"Connected to violation logs for job {job_id}",
            "frame": 0
        })

        # Keep connection alive and listen for client messages (e.g., ping)
        while True:
            try:
                # Wait for messages from client (or just keep alive)
                data = await websocket.receive_text()

                # Handle ping/pong or other client messages
                if data == "ping":
                    await websocket.send_json({
                        "timestamp": time.strftime("%H:%M:%S"),
                        "level": "info",
                        "zone": "System",
                        "message": "pong",
                        "frame": 0
                    })
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
                break

    finally:
        # Remove from connections list
        if job_id in websocket_connections:
            if websocket in websocket_connections[job_id]:
                websocket_connections[job_id].remove(websocket)

            # Clean up empty lists
            if not websocket_connections[job_id]:
                del websocket_connections[job_id]

        logger.info(f"WebSocket client disconnected for job {job_id}")


@app.post("/test_stream")
async def test_stream_connection(request: dict):
    """
    Test stream connection without starting full processing
    Useful for debugging stream issues

    Body: {"stream_url": "udp://127.0.0.1:1905"}
    """
    stream_url = request.get("stream_url")

    if not stream_url:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "message": "stream_url is required in request body"
            }
        )

    try:
        from utils.stream_reader import StreamReader

        logger.info(f"Testing stream connection: {stream_url}")

        # Test stream connection
        stream_reader = StreamReader(stream_url, use_ffmpeg_for_udp=True)
        props = stream_reader.get_properties()

        # Try to read a few frames
        frames_read = 0
        for i in range(3):
            ret, frame = stream_reader.read()
            if ret and frame is not None:
                frames_read += 1
            time.sleep(0.1)

        stream_reader.release()

        return JSONResponse(content={
            "status": "success",
            "stream_url": stream_url,
            "properties": props,
            "frames_read": frames_read,
            "message": f"Stream test completed. Read {frames_read}/3 frames successfully."
        })

    except Exception as e:
        logger.error(f"Stream test failed: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "stream_url": stream_url,
                "error": str(e),
                "message": "Stream connection failed. Check stream URL and ensure stream is broadcasting."
            }
        )


@app.post("/detect")
async def detect_and_track(
    background_tasks: BackgroundTasks,
    video: Optional[UploadFile] = File(None),
    stream_url: Optional[str] = Form(None),
    config_path: Optional[str] = Form(None),
    similarity_threshold: float = Form(0.8),
    model_type: Optional[str] = Form(None),
    conf_thresh: Optional[float] = Form(None),
    track_thresh: Optional[float] = Form(None),
    face_conf_thresh: Optional[float] = Form(None),
    zone_config: Optional[UploadFile] = File(None),
    iou_threshold: float = Form(0.6),
    zone_opacity: float = Form(0.3),
    max_frames: Optional[int] = Form(None),
    max_duration_seconds: Optional[int] = Form(None),
    alert_threshold: float = Form(0),
    zone_workers: Optional[int] = Form(None)
):
    """
    Detect, track, and re-identify persons in video file or stream with optional zone monitoring

    This unified endpoint handles both:
    - Video file upload (provide 'video' parameter)
    - Stream URL (provide 'stream_url' parameter)

    Args:
        video: Video file to process (mutually exclusive with stream_url)
        stream_url: Stream URL like 'udp://127.0.0.1:1905' or 'rtsp://camera_ip/stream' (mutually exclusive with video)
        config_path: Optional path to custom config file
        similarity_threshold: Cosine similarity threshold for ReID (default: 0.8)
        model_type: Detection model type - 'mot17' or 'yolox' (default: from config)
        conf_thresh: Detection confidence threshold 0-1 (default: from config)
        track_thresh: Tracking confidence threshold 0-1 (default: from config)
        face_conf_thresh: Face detection confidence threshold 0-1 (default: from config)
        zone_config: Optional zone configuration YAML file
        iou_threshold: Zone IoP threshold 0-1 (default: 0.6 = 60%)
        zone_opacity: Zone border thickness factor 0-1 (default: 0.3 = 3px)
        max_frames: Maximum frames to process (None for unlimited, mainly for streams)
        max_duration_seconds: Maximum duration in seconds (None for unlimited, mainly for streams)
        alert_threshold: Time threshold (seconds) before showing alert (default: 0)
        zone_workers: Number of worker processes for zone monitoring (default: auto)

    Returns:
        Job ID for tracking the detection process
    """
    try:
        # Validate input: must provide either video or stream_url, but not both
        if video is None and stream_url is None:
            raise HTTPException(status_code=400, detail="Must provide either 'video' file or 'stream_url'")
        if video is not None and stream_url is not None:
            raise HTTPException(status_code=400, detail="Cannot provide both 'video' and 'stream_url'. Choose one.")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Determine if this is a stream or video file
        is_stream = stream_url is not None

        # Handle video file upload or stream URL
        if is_stream:
            # Stream processing
            video_path_str = stream_url
            logger.info(f"Processing stream: {stream_url}")
        else:
            # Video file processing
            video_filename = f"{job_id}_{video.filename}"
            video_path = UPLOAD_DIR / video_filename
            with open(video_path, "wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
            video_path_str = str(video_path)
            logger.info(f"Processing video file: {video_path_str}")

        # Save zone config if provided
        zone_config_path = _save_zone_config(zone_config, job_id)

        # Setup output paths
        output_suffix = "stream" if is_stream else "video"
        output_video = str(OUTPUT_DIR / "videos" / f"{job_id}_{output_suffix}_output.mp4")
        output_csv = str(OUTPUT_DIR / "csv" / f"{job_id}_{output_suffix}_tracking.csv")
        output_log = str(OUTPUT_DIR / "logs" / f"{job_id}_{output_suffix}_detection.log")

        # Create output directories
        _ensure_output_directories()

        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": f"{'Stream' if is_stream else 'Video'} detection job queued for processing",
            "video_path": video_path_str,
            "zone_monitoring": zone_config_path is not None,
            "is_stream": is_stream
        }

        # Add stream-specific metadata
        if is_stream:
            jobs[job_id]["max_frames"] = max_frames
            jobs[job_id]["max_duration_seconds"] = max_duration_seconds

        # Add background task
        background_tasks.add_task(
            process_detection,
            job_id=job_id,
            video_path=video_path_str,
            output_video=output_video,
            output_csv=output_csv,
            output_log=output_log,
            config_path=config_path,
            similarity_threshold=similarity_threshold,
            model_type=model_type,
            conf_thresh=conf_thresh,
            track_thresh=track_thresh,
            face_conf_thresh=face_conf_thresh,
            zone_config_path=str(zone_config_path) if zone_config_path else None,
            iou_threshold=iou_threshold,
            zone_opacity=zone_opacity,
            max_frames=max_frames,
            max_duration_seconds=max_duration_seconds,
            alert_threshold=alert_threshold,
            zone_workers=zone_workers
        )

        # Prepare response
        response_data = {
            "job_id": job_id,
            "status": "pending",
            "message": f"{'Stream' if is_stream else 'Video'} detection job started"
        }

        if is_stream:
            response_data["stream_url"] = stream_url
            response_data["max_frames"] = max_frames
            response_data["max_duration_seconds"] = max_duration_seconds

        return JSONResponse(content=response_data)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Removed: /detect_stream endpoint - merged into unified /detect endpoint
# The /detect endpoint now handles both video files and stream URLs


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a detection job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**jobs[job_id])


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get real-time progress of a detection job (supports both single and multi-camera)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id not in progress_data:
        return ProgressStatus(
            job_id=job_id,
            status=jobs[job_id]["status"],
            current_frame=0,
            total_frames=0,
            progress_percent=0.0,
            tracks=[],
            violations=[]
        )

    prog = progress_data[job_id]

    # Check if multi-camera mode
    if "cameras" in prog:
        # Multi-camera: aggregate progress
        cameras_data = prog["cameras"]
        num_cameras = prog.get("num_streams", len(cameras_data))

        # Calculate average frame across all cameras
        total_frames_processed = sum(cam.get("current_frame", 0) for cam in cameras_data.values())
        avg_frame = total_frames_processed // num_cameras if num_cameras > 0 else 0

        # Aggregate tracks from all cameras
        all_tracks = []
        for cam_id, cam_data in cameras_data.items():
            for track in cam_data.get("tracks", []):
                track_copy = track.copy()
                track_copy["camera_id"] = cam_id
                all_tracks.append(track_copy)

        return ProgressStatus(
            job_id=job_id,
            status=jobs[job_id]["status"],
            current_frame=avg_frame,
            total_frames=0,  # Streams don't have total frames
            progress_percent=0.0,
            tracks=all_tracks,
            violations=prog.get("violations", []),
            message=f"Processing {num_cameras} cameras (avg frame: {avg_frame})",
            cameras=cameras_data  # Include per-camera data
        )
    else:
        # Single-camera: backward compatible
        current_frame = prog.get("current_frame", 0)
        total_frames = prog.get("total_frames", 0)
        progress_percent = (current_frame / total_frames * 100) if total_frames > 0 else 0

        return ProgressStatus(
            job_id=job_id,
            status=jobs[job_id]["status"],
            current_frame=current_frame,
            total_frames=total_frames,
            progress_percent=progress_percent,
            tracks=prog.get("tracks", []),
            violations=prog.get("violations", []),
            message=f"Processing frame {current_frame}/{total_frames}"
        )


@app.get("/download/video/{job_id}")
async def download_video(job_id: str):
    """Download the output video"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    video_path = Path(jobs[job_id]["output_video"])
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename=f"{job_id}_output.mp4"
    )


@app.get("/download/csv/{job_id}")
async def download_csv(job_id: str):
    """Download the tracking CSV"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    csv_path = Path(jobs[job_id]["output_csv"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="CSV file not found")
    
    return FileResponse(
        path=csv_path,
        media_type="text/csv",
        filename=f"{job_id}_tracking.csv"
    )


@app.get("/download/log/{job_id}")
async def download_log(job_id: str):
    """Download the detection log"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    log_path = Path(jobs[job_id]["output_log"])
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    return FileResponse(
        path=log_path,
        media_type="text/plain",
        filename=f"{job_id}_detection.log"
    )


@app.get("/download/json/{job_id}")
async def download_json(job_id: str):
    """Download the zone monitoring JSON report"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    if "output_json" not in jobs[job_id]:
        raise HTTPException(status_code=404, detail="Zone monitoring was not enabled for this job")

    json_path = Path(jobs[job_id]["output_json"])
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="JSON report file not found")

    return FileResponse(
        path=json_path,
        media_type="application/json",
        filename=f"{job_id}_zones.json"
    )


@app.get("/download/zip/{job_id}")
async def download_all_as_zip(job_id: str):
    """Download all outputs as a ZIP file (supports both single-stream and multi-stream)"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    # Determine output directory and ZIP filename
    is_multi_stream = jobs[job_id].get("is_multi_stream", False)

    if is_multi_stream:
        # Multi-stream: use multi_stream_dir
        if "multi_stream_dir" not in jobs[job_id]:
            raise HTTPException(status_code=404, detail="Multi-stream output not found.")

        multi_stream_dirname = jobs[job_id]["multi_stream_dir"]
        output_dir = OUTPUT_DIR / multi_stream_dirname
        zip_filename = f"{multi_stream_dirname}_results.zip"
    else:
        # Single-stream: create ZIP from individual output files
        output_dir = OUTPUT_DIR
        # Generate timestamp for single-stream ZIP
        from datetime import datetime, timezone, timedelta
        tz_hcm = timezone(timedelta(hours=7))
        timestamp = datetime.now(tz_hcm).strftime("%Y-%m-%d-%H-%M")
        zip_filename = f"single_stream_{timestamp}_results.zip"

    if not output_dir.exists():
        raise HTTPException(status_code=404, detail=f"Output directory not found: {output_dir}")

    # Create a temporary ZIP file
    import tempfile
    import zipfile

    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip_path = temp_zip.name
    temp_zip.close()

    try:
        with zipfile.ZipFile(temp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if is_multi_stream:
                # Multi-stream: Add all files from multi_stream directory
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        # Create relative path for ZIP archive
                        arcname = file_path.relative_to(output_dir.parent)
                        zipf.write(file_path, arcname=arcname)
                        logger.info(f"Added to ZIP: {arcname}")
            else:
                # Single-stream: Add individual output files for this job
                job_files = [
                    jobs[job_id].get("output_video"),
                    jobs[job_id].get("output_csv"),
                    jobs[job_id].get("zone_report")
                ]

                for file_path_str in job_files:
                    if file_path_str:
                        file_path = Path(file_path_str)
                        if file_path.exists() and file_path.is_file():
                            # Use just the filename in ZIP (no subdirectories)
                            zipf.write(file_path, arcname=file_path.name)
                            logger.info(f"Added to ZIP: {file_path.name}")

        # Return the ZIP file
        return FileResponse(
            path=temp_zip_path,
            media_type="application/zip",
            filename=zip_filename,
            background=BackgroundTask(lambda: os.unlink(temp_zip_path))  # Clean up temp file after sending
        )

    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_zip_path):
            os.unlink(temp_zip_path)
        logger.error(f"Failed to create ZIP: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create ZIP file: {str(e)}")


@app.post("/cancel/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running detection job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_status = jobs[job_id]["status"]

    if job_status in ["completed", "failed"]:
        return JSONResponse(content={
            "job_id": job_id,
            "message": f"Job already {job_status}, cannot cancel"
        })

    if job_status == "cancelled":
        return JSONResponse(content={
            "job_id": job_id,
            "message": "Job already cancelled"
        })

    # Set cancellation flag
    if job_id in cancellation_flags:
        cancellation_flags[job_id].set()
        logger.info(f"Cancellation requested for job {job_id}")

    # Update job status
    jobs[job_id]["status"] = "cancelled"
    jobs[job_id]["message"] = "Job cancelled by user"

    return JSONResponse(content={
        "job_id": job_id,
        "status": "cancelled",
        "message": "Job cancellation requested"
    })


@app.get("/zones/from-db")
async def get_zones_from_database(camera_name: str = "camera_1"):
    """
    Get zone configuration from database

    Args:
        camera_name: Camera identifier (default: "camera_1")

    Returns:
        Zone configuration in YAML-compatible format
    """
    try:
        from services.zone_db_loader import get_zone_config_dict

        zone_config = get_zone_config_dict(camera_name)

        if not zone_config:
            raise HTTPException(
                status_code=404,
                detail="No zones found in database or database connection failed"
            )

        return JSONResponse(content=zone_config)

    except Exception as e:
        logger.error(f"Error loading zones from database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """
    Get the status of a detection job

    Args:
        job_id: Job ID to check

    Returns:
        Job status and progress information
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_info = jobs[job_id].copy()

    # Add progress information if available
    if job_id in progress_data:
        job_info["progress"] = progress_data[job_id]

    return JSONResponse(content=job_info)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

