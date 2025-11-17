#!/usr/bin/env python3
"""
Extract Service API
FastAPI service for extracting individual object videos from tracking results
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import shutil
import zipfile
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from loguru import logger
from scripts.extract_objects import extract_objects_from_video

app = FastAPI(title="Extract Service", version="1.0.0")

# Storage paths
UPLOAD_DIR = Path("/app/data/uploads")
OUTPUT_DIR = Path("/app/outputs/extracted_objects")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Job storage
jobs = {}


@app.on_event("startup")
async def startup_event():
    """Ensure output directories exist at service startup"""
    try:
        # Ensure output directories exist (important for Docker volume mounts)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Extract Service directories ready: {OUTPUT_DIR}")
    except Exception as e:
        logger.error(f"❌ Failed to create directories: {e}")


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    message: Optional[str] = None
    output_dir: Optional[str] = None
    error: Optional[str] = None


def process_extraction(job_id: str, video_path: str, output_dir: str, 
                       model_type: str, padding: int, conf_thresh: float,
                       track_thresh: float, min_frames: int):
    """Background task to process video extraction"""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting extraction job {job_id}")
        
        extract_objects_from_video(
            video_path=video_path,
            output_dir=output_dir,
            model_type=model_type,
            padding=padding,
            conf_thresh=conf_thresh,
            track_thresh=track_thresh,
            min_frames=min_frames
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Extraction completed successfully"
        jobs[job_id]["output_dir"] = output_dir
        logger.info(f"Extraction job {job_id} completed")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Extraction job {job_id} failed: {e}")


@app.get("/")
async def root():
    return {"service": "Extract Service", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/extract")
async def extract_video(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    model_type: str = Form("mot17"),
    padding: int = Form(10),
    conf_thresh: float = Form(0.6),
    track_thresh: float = Form(0.5),
    min_frames: int = Form(10)
):
    """
    Extract individual object videos from uploaded video
    
    Args:
        video: Video file to process
        model_type: Model type ('mot17' or 'yolox')
        padding: Padding pixels around bounding box
        conf_thresh: Detection confidence threshold
        track_thresh: Tracking confidence threshold
        min_frames: Minimum frames required to save an object
    
    Returns:
        Job ID for tracking the extraction process
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded video
        video_filename = f"{job_id}_{video.filename}"
        video_path = UPLOAD_DIR / video_filename
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Create output directory for this job
        job_output_dir = OUTPUT_DIR / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job queued for processing",
            "video_path": str(video_path),
            "output_dir": str(job_output_dir)
        }
        
        # Add background task
        background_tasks.add_task(
            process_extraction,
            job_id=job_id,
            video_path=str(video_path),
            output_dir=str(job_output_dir),
            model_type=model_type,
            padding=padding,
            conf_thresh=conf_thresh,
            track_thresh=track_thresh,
            min_frames=min_frames
        )
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "pending",
            "message": "Extraction job started"
        })
        
    except Exception as e:
        logger.error(f"Error starting extraction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of an extraction job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])


@app.get("/results/{job_id}")
async def list_results(job_id: str):
    """List extracted object videos for a job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    if jobs[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed yet")

    output_dir = Path(jobs[job_id]["output_dir"])
    if not output_dir.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    # List all video files recursively (extract_objects creates subdirectory)
    video_files = list(output_dir.glob("**/*.mp4"))

    return {
        "job_id": job_id,
        "total_objects": len(video_files),
        "files": [f.name for f in video_files]
    }


# Specific routes MUST come before generic routes in FastAPI
@app.get("/download/zip/{job_id}")
async def download_all_as_zip(job_id: str):
    """Download all extracted object videos as a ZIP file"""
    # Check filesystem directly (works even after service restart)
    output_base = Path("/app/outputs/extracted_objects") / job_id

    if not output_base.exists():
        raise HTTPException(status_code=404, detail="Job output not found")

    # Get all video files recursively
    video_files = list(output_base.glob("**/*.mp4"))

    if not video_files:
        raise HTTPException(status_code=404, detail="No video files found")

    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for video_file in video_files:
            # Add file to ZIP with just the filename (flatten structure)
            zip_file.write(video_file, arcname=video_file.name)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}_extracted_objects.zip"
        }
    )


@app.get("/download/log/{job_id}")
async def download_log(job_id: str):
    """Download extraction log file"""
    # Check filesystem directly (works even after service restart)
    output_base = Path("/app/outputs/extracted_objects") / job_id

    if not output_base.exists():
        raise HTTPException(status_code=404, detail="Job output not found")

    # Get job info from memory if available, otherwise use defaults
    job_info = jobs.get(job_id, {})
    status = job_info.get('status', 'completed' if output_base.exists() else 'unknown')
    video_path = job_info.get('video_path', 'N/A')
    message = job_info.get('message', 'Extraction completed')
    error = job_info.get('error', 'None')

    # Create log content
    log_content = f"""Extraction Job Log
==================
Job ID: {job_id}
Status: {status}
Video Path: {video_path}
Output Directory: {output_base}
Message: {message}
Error: {error}

Results:
--------
"""

    # Add results from filesystem
    video_files = list(output_base.glob("**/*.mp4"))
    log_content += f"Total Objects Extracted: {len(video_files)}\n\n"
    log_content += "Extracted Files:\n"
    for video_file in sorted(video_files):
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        log_content += f"  - {video_file.name} ({file_size:.2f} MB)\n"

    # Return as downloadable text file
    return StreamingResponse(
        io.BytesIO(log_content.encode()),
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename={job_id}_extraction.log"
        }
    )


# Generic route MUST come after specific routes
@app.get("/download/{job_id}/{filename}")
async def download_result(job_id: str, filename: str):
    """Download a specific extracted object video"""
    # Check filesystem directly (works even after service restart)
    output_base = Path("/app/outputs/extracted_objects") / job_id

    if not output_base.exists():
        raise HTTPException(status_code=404, detail="Job output not found")

    # Search for file recursively (extract_objects creates subdirectory)
    matching_files = list(output_base.glob(f"**/{filename}"))

    if not matching_files:
        raise HTTPException(status_code=404, detail="File not found")

    file_path = matching_files[0]  # Use first match

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)

