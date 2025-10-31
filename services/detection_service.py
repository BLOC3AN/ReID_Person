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

app = FastAPI(title="Detection Service", version="1.0.0")

# Storage paths
UPLOAD_DIR = Path("/app/data/uploads")
OUTPUT_DIR = Path("/app/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job storage
jobs = {}
# Progress tracking
progress_data = {}  # {job_id: {"current_frame": int, "total_frames": int, "tracks": [...]}}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
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
                      zone_config_path: Optional[str] = None, iou_threshold: float = 0.6,
                      zone_opacity: float = 0.15):
    """Background task to process detection and tracking with optional zone monitoring"""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting detection job {job_id}")

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
                max_frames=None,
                progress_callback=lambda frame_id, tracks: _update_progress(job_id, frame_id, tracks)
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

            # Initialize pipeline
            pipeline = PersonReIDPipeline(config_path=config_path)

            # Override config parameters if provided
            if model_type is not None:
                pipeline.config['detection']['model_type'] = model_type

            if conf_thresh is not None:
                pipeline.config['detection']['conf_threshold'] = conf_thresh

            if track_thresh is not None:
                pipeline.config['tracking']['track_thresh'] = track_thresh

            # Components will be initialized automatically in process_video()

            # Get total frames from video
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
            progress_callback=lambda frame_id, tracks: _update_progress(job_id, frame_id, tracks)
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Detection and tracking completed successfully"
        jobs[job_id]["output_video"] = output_video
        jobs[job_id]["output_csv"] = output_csv
        jobs[job_id]["output_log"] = output_log
        logger.info(f"Detection job {job_id} completed")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Detection job {job_id} failed: {e}")
    finally:
        # Cleanup progress data after some time
        if job_id in progress_data:
            progress_data[job_id]["cleanup_time"] = time.time()


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


@app.post("/detect")
async def detect_and_track(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    config_path: Optional[str] = Form(None),
    similarity_threshold: float = Form(0.8),
    model_type: Optional[str] = Form(None),
    conf_thresh: Optional[float] = Form(None),
    track_thresh: Optional[float] = Form(None),
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
            zone_config_filename = f"{job_id}_zones.yaml"
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)

