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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import uuid
from loguru import logger
import threading
import time

# Import modules
from scripts.detect_and_track import PersonReIDPipeline
from scripts.zone_monitor import process_video_with_zones
from core.preloaded_manager import preloaded_manager

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
progress_data = {}  # {job_id: {"current_frame": int, "total_frames": int, "tracks": [...]}}
# Cancellation flags
cancellation_flags = {}  # {job_id: threading.Event()}


@app.on_event("startup")
async def startup_event():
    """Initialize pre-loaded components at service startup"""
    logger.info("🚀 Detection Service starting up...")
    try:
        # Ensure output directories exist (important for Docker volume mounts)
        (OUTPUT_DIR / "videos").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
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
    message: Optional[str] = None


def process_detection(job_id: str, video_path: str, output_video: str,
                      output_csv: str, output_log: str, config_path: Optional[str],
                      similarity_threshold: float = 0.8, model_type: Optional[str] = None,
                      conf_thresh: Optional[float] = None, track_thresh: Optional[float] = None,
                      face_conf_thresh: Optional[float] = None,
                      zone_config_path: Optional[str] = None, iou_threshold: float = 0.6,
                      zone_opacity: float = 0.15, max_frames: Optional[int] = None,
                      max_duration_seconds: Optional[int] = None):
    """Background task to process detection and tracking with optional zone monitoring"""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting detection job {job_id}")

        # Initialize cancellation flag
        cancellation_flags[job_id] = threading.Event()

        # Initialize progress tracking
        progress_data[job_id] = {
            "current_frame": 0,
            "total_frames": 0,
            "tracks": [],
            "last_update": time.time()
        }

        # Check if zone monitoring is enabled
        use_zones = zone_config_path is not None and Path(zone_config_path).exists()

        if use_zones:
            logger.info(f"Zone monitoring enabled: {zone_config_path}")
            # Use zone monitoring pipeline
            from scripts.zone_monitor import process_video_with_zones

            # Get total frames
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
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
                cancellation_flag=cancellation_flags.get(job_id)
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

            # Update job outputs
            jobs[job_id]["output_video"] = output_video
            jobs[job_id]["output_csv"] = output_csv
            jobs[job_id]["output_log"] = output_log
            jobs[job_id]["output_json"] = output_json

        else:
            # Standard detection without zones
            logger.info("Standard detection (no zone monitoring)")

            # Initialize pipeline with pre-loaded components
            pipeline = PersonReIDPipeline(config_path=config_path, use_preloaded=True)

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

            # Get total frames from video (will be 0 for streams)
            import cv2
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
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
            jobs[job_id]["output_video"] = output_video
            jobs[job_id]["output_csv"] = output_csv
            jobs[job_id]["output_log"] = output_log
            logger.info(f"Detection job {job_id} completed")

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


def _update_progress(job_id: str, frame_id: int, tracks: List[dict]):
    """Update progress data for a job"""
    if job_id in progress_data:
        progress_data[job_id]["current_frame"] = frame_id
        progress_data[job_id]["tracks"] = tracks
        progress_data[job_id]["last_update"] = time.time()


@app.get("/")
async def root():
    return {"service": "Detection Service", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
    video: UploadFile = File(...),
    config_path: Optional[str] = Form(None),
    similarity_threshold: float = Form(0.8),
    model_type: Optional[str] = Form(None),
    conf_thresh: Optional[float] = Form(None),
    track_thresh: Optional[float] = Form(None),
    face_conf_thresh: Optional[float] = Form(None),
    zone_config: Optional[UploadFile] = File(None),
    iou_threshold: float = Form(0.6),
    zone_opacity: float = Form(0.15)
):
    """
    Detect, track, and re-identify persons in video with optional zone monitoring

    Args:
        video: Video file to process
        config_path: Optional path to custom config file
        similarity_threshold: Cosine similarity threshold for ReID (default: 0.8)
        model_type: Detection model type - 'mot17' or 'yolox' (default: from config)
        conf_thresh: Detection confidence threshold 0-1 (default: from config)
        track_thresh: Tracking confidence threshold 0-1 (default: from config)
        face_conf_thresh: Face detection confidence threshold 0-1 (default: from config)
        zone_config: Optional zone configuration YAML file
        iou_threshold: Zone IoP threshold 0-1 (default: 0.6 = 60%)
        zone_opacity: Zone fill opacity 0-1 (default: 0.15 = 15%)

    Returns:
        Job ID for tracking the detection process
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Save uploaded video
        video_filename = f"{job_id}_{video.filename}"
        video_path = UPLOAD_DIR / video_filename

        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Save zone config if provided
        zone_config_path = None
        if zone_config is not None:
            # Preserve original file extension (.yaml, .yml, or .json)
            original_ext = Path(zone_config.filename).suffix
            if original_ext not in ['.yaml', '.yml', '.json']:
                original_ext = '.yaml'  # Default to .yaml if unknown
            zone_config_filename = f"{job_id}_zones{original_ext}"
            zone_config_path = UPLOAD_DIR / zone_config_filename

            with open(zone_config_path, "wb") as buffer:
                shutil.copyfileobj(zone_config.file, buffer)

            logger.info(f"Zone config saved: {zone_config_path}")

        # Setup output paths
        output_video = str(OUTPUT_DIR / "videos" / f"{job_id}_output.mp4")
        output_csv = str(OUTPUT_DIR / "csv" / f"{job_id}_tracking.csv")
        output_log = str(OUTPUT_DIR / "logs" / f"{job_id}_detection.log")

        # Create output directories
        (OUTPUT_DIR / "videos").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job queued for processing",
            "video_path": str(video_path),
            "zone_monitoring": zone_config_path is not None
        }

        # Add background task
        background_tasks.add_task(
            process_detection,
            job_id=job_id,
            video_path=str(video_path),
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
            zone_opacity=zone_opacity
        )
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "pending",
            "message": "Detection job started"
        })
        
    except Exception as e:
        logger.error(f"Error starting detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/detect_stream")
async def detect_and_track_stream(
    background_tasks: BackgroundTasks,
    stream_url: str = Form(...),
    config_path: Optional[str] = Form(None),
    similarity_threshold: float = Form(0.8),
    model_type: Optional[str] = Form(None),
    conf_thresh: Optional[float] = Form(None),
    track_thresh: Optional[float] = Form(None),
    face_conf_thresh: Optional[float] = Form(None),
    zone_config: Optional[UploadFile] = File(None),
    iou_threshold: float = Form(0.6),
    zone_opacity: float = Form(0.15),
    max_frames: Optional[int] = Form(None),
    max_duration_seconds: Optional[int] = Form(None)
):
    """
    Detect, track, and re-identify persons from video stream (UDP/RTSP)

    Args:
        stream_url: Stream URL (e.g., 'udp://127.0.0.1:1905', 'rtsp://camera_ip/stream')
        config_path: Optional path to custom config file
        similarity_threshold: Cosine similarity threshold for ReID (default: 0.8)
        model_type: Detection model type - 'mot17' or 'yolox' (default: from config)
        conf_thresh: Detection confidence threshold 0-1 (default: from config)
        track_thresh: Tracking confidence threshold 0-1 (default: from config)
        face_conf_thresh: Face detection confidence threshold 0-1 (default: from config)
        zone_config: Optional zone configuration YAML file
        iou_threshold: Zone IoP threshold 0-1 (default: 0.6 = 60%)
        zone_opacity: Zone fill opacity 0-1 (default: 0.15 = 15%)
        max_frames: Maximum frames to process (None for unlimited)
        max_duration_seconds: Maximum duration in seconds (None for unlimited)

    Returns:
        Job ID for tracking the detection process
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())

        # Save zone config if provided
        zone_config_path = None
        if zone_config is not None:
            # Preserve original file extension (.yaml, .yml, or .json)
            original_ext = Path(zone_config.filename).suffix
            if original_ext not in ['.yaml', '.yml', '.json']:
                original_ext = '.yaml'  # Default to .yaml if unknown
            zone_config_filename = f"{job_id}_zones{original_ext}"
            zone_config_path = UPLOAD_DIR / zone_config_filename

            with open(zone_config_path, "wb") as buffer:
                shutil.copyfileobj(zone_config.file, buffer)

            logger.info(f"Zone config saved: {zone_config_path}")

        # Setup output paths
        output_video = str(OUTPUT_DIR / "videos" / f"{job_id}_stream_output.mp4")
        output_csv = str(OUTPUT_DIR / "csv" / f"{job_id}_stream_tracking.csv")
        output_log = str(OUTPUT_DIR / "logs" / f"{job_id}_stream_detection.log")

        # Create output directories
        (OUTPUT_DIR / "videos").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "csv").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Stream detection job queued for processing",
            "video_path": stream_url,
            "zone_monitoring": zone_config_path is not None,
            "is_stream": True,
            "max_frames": max_frames,
            "max_duration_seconds": max_duration_seconds
        }

        # Add background task
        background_tasks.add_task(
            process_detection,
            job_id=job_id,
            video_path=stream_url,  # Pass stream URL directly
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
            max_duration_seconds=max_duration_seconds
        )

        return JSONResponse(content={
            "job_id": job_id,
            "status": "pending",
            "message": "Stream detection job started",
            "stream_url": stream_url,
            "max_frames": max_frames,
            "max_duration_seconds": max_duration_seconds
        })

    except Exception as e:
        logger.error(f"Error starting stream detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a detection job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobStatus(**jobs[job_id])


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    """Get real-time progress of a detection job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_id not in progress_data:
        return ProgressStatus(
            job_id=job_id,
            status=jobs[job_id]["status"],
            current_frame=0,
            total_frames=0,
            progress_percent=0.0,
            tracks=[]
        )

    prog = progress_data[job_id]
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

