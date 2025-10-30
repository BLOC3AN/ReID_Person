#!/usr/bin/env python3
"""
Register Service API
FastAPI service for registering persons into the vector database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import shutil
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uuid
from loguru import logger
from scripts.register_mot17 import register_person_mot17

app = FastAPI(title="Register Service", version="1.0.0")

# Storage paths
UPLOAD_DIR = Path("/app/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Job storage
jobs = {}


class JobStatus(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    message: Optional[str] = None
    person_name: Optional[str] = None
    global_id: Optional[int] = None
    error: Optional[str] = None


def process_registration(job_id: str, video_path: str, person_name: str, 
                         global_id: int, sample_rate: int, delete_existing: bool):
    """Background task to process person registration"""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting registration job {job_id} for {person_name}")
        
        register_person_mot17(
            video_path=video_path,
            person_name=person_name,
            global_id=global_id,
            sample_rate=sample_rate,
            delete_existing=delete_existing
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = f"Person {person_name} registered successfully"
        logger.info(f"Registration job {job_id} completed")
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Registration job {job_id} failed: {e}")


@app.get("/")
async def root():
    return {"service": "Register Service", "status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/register")
async def register_person(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    person_name: str = Form(...),
    global_id: int = Form(...),
    sample_rate: int = Form(5),
    delete_existing: bool = Form(False)
):
    """
    Register a person into the vector database
    
    Args:
        video: Video file containing the person
        person_name: Name of the person to register
        global_id: Global ID for the person (unique identifier)
        sample_rate: Extract 1 frame every N frames (default: 5)
        delete_existing: Delete existing collection before registering
    
    Returns:
        Job ID for tracking the registration process
    """
    try:
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded video
        video_filename = f"{job_id}_{video.filename}"
        video_path = UPLOAD_DIR / video_filename
        
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job queued for processing",
            "person_name": person_name,
            "global_id": global_id,
            "video_path": str(video_path)
        }
        
        # Add background task
        background_tasks.add_task(
            process_registration,
            job_id=job_id,
            video_path=str(video_path),
            person_name=person_name,
            global_id=global_id,
            sample_rate=sample_rate,
            delete_existing=delete_existing
        )
        
        return JSONResponse(content={
            "job_id": job_id,
            "status": "pending",
            "message": f"Registration job started for {person_name}"
        })
        
    except Exception as e:
        logger.error(f"Error starting registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get status of a registration job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**jobs[job_id])


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Delete uploaded video
    video_path = Path(jobs[job_id].get("video_path", ""))
    if video_path.exists():
        video_path.unlink()
    
    # Remove job from storage
    del jobs[job_id]
    
    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

