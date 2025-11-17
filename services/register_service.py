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
from typing import Optional, List
import uuid
from loguru import logger
from scripts.register_mot17 import register_person_mot17, register_person_from_images
from services.preload_models import preload_models
from core.preloaded_manager import preloaded_manager

app = FastAPI(title="Register Service", version="1.0.0")

# Storage paths
UPLOAD_DIR = Path("/app/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Preload models at startup
@app.on_event("startup")
async def startup_event():
    """Preload models when service starts"""
    try:
        logger.info("🚀 Starting Register Service...")
        # Ensure upload directory exists (important for Docker volume mounts)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Upload directory ready: {UPLOAD_DIR}")
        preload_models()
        logger.info("✅ Register Service ready!")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise

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
            delete_existing=delete_existing,
            detector=preloaded_manager.detector,
            extractor=preloaded_manager.extractor
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = f"Person {person_name} registered successfully"
        logger.info(f"Registration job {job_id} completed")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Registration job {job_id} failed: {e}")


def process_image_registration(job_id: str, image_paths: List[str], person_name: str,
                                global_id: int, delete_existing: bool):
    """Background task to process person registration from images"""
    try:
        jobs[job_id]["status"] = "processing"
        logger.info(f"Starting image registration job {job_id} for {person_name}")

        register_person_from_images(
            image_paths=image_paths,
            person_name=person_name,
            global_id=global_id,
            delete_existing=delete_existing,
            detector=preloaded_manager.detector,
            extractor=preloaded_manager.extractor
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = f"Person {person_name} registered successfully from {len(image_paths)} images"
        logger.info(f"Image registration job {job_id} completed")

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        logger.error(f"Image registration job {job_id} failed: {e}")


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


@app.post("/register-batch")
async def register_batch(
    background_tasks: BackgroundTasks,
    videos: List[UploadFile] = File(...),
    person_name: str = Form(...),
    global_id: int = Form(...),
    sample_rate: int = Form(5),
    delete_existing: bool = Form(False)
):
    """
    Register a person with multiple videos into the vector database

    Args:
        videos: Multiple video files containing the person
        person_name: Name of the person to register
        global_id: Global ID for the person (unique identifier)
        sample_rate: Extract 1 frame every N frames (default: 5)
        delete_existing: Delete existing collection before registering

    Returns:
        List of Job IDs for tracking the registration process
    """
    try:
        if not videos:
            raise HTTPException(status_code=400, detail="No videos provided")

        job_ids = []

        for video in videos:
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
                "video_path": str(video_path),
                "video_filename": video.filename
            }

            # Add background task
            background_tasks.add_task(
                process_registration,
                job_id=job_id,
                video_path=str(video_path),
                person_name=person_name,
                global_id=global_id,
                sample_rate=sample_rate,
                delete_existing=delete_existing and (job_ids == [])  # Only delete on first video
            )

            job_ids.append(job_id)

        return JSONResponse(content={
            "job_ids": job_ids,
            "total_videos": len(job_ids),
            "status": "pending",
            "message": f"Registration jobs started for {person_name} with {len(job_ids)} video(s)"
        })

    except Exception as e:
        logger.error(f"Error starting batch registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/status-batch")
async def get_batch_status(job_ids: str = None):
    """
    Get status of multiple registration jobs

    Args:
        job_ids: Comma-separated list of job IDs

    Returns:
        Status of all requested jobs
    """
    if not job_ids:
        raise HTTPException(status_code=400, detail="No job IDs provided")

    job_id_list = [jid.strip() for jid in job_ids.split(",")]
    statuses = []

    for job_id in job_id_list:
        if job_id in jobs:
            statuses.append(JobStatus(**jobs[job_id]))
        else:
            statuses.append({
                "job_id": job_id,
                "status": "not_found",
                "message": "Job not found"
            })

    return {
        "total_jobs": len(job_id_list),
        "jobs": statuses
    }


@app.post("/register-images")
async def register_person_images(
    background_tasks: BackgroundTasks,
    images: List[UploadFile] = File(...),
    person_name: str = Form(...),
    global_id: int = Form(...),
    delete_existing: bool = Form(False)
):
    """
    Register a person from images into the vector database

    Args:
        images: Image files containing the person
        person_name: Name of the person to register
        global_id: Global ID for the person (unique identifier)
        delete_existing: Delete existing collection before registering

    Returns:
        Job ID for tracking the registration process
    """
    try:
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Save uploaded images
        image_paths = []
        for img in images:
            img_filename = f"{job_id}_{img.filename}"
            img_path = UPLOAD_DIR / img_filename

            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)

            image_paths.append(str(img_path))

        # Initialize job status
        jobs[job_id] = {
            "job_id": job_id,
            "status": "pending",
            "message": "Job queued for processing",
            "person_name": person_name,
            "global_id": global_id,
            "image_paths": image_paths,
            "num_images": len(image_paths)
        }

        # Add background task
        background_tasks.add_task(
            process_image_registration,
            job_id=job_id,
            image_paths=image_paths,
            person_name=person_name,
            global_id=global_id,
            delete_existing=delete_existing
        )

        return JSONResponse(content={
            "job_id": job_id,
            "status": "pending",
            "message": f"Image registration job started for {person_name} with {len(image_paths)} image(s)"
        })

    except Exception as e:
        logger.error(f"Error starting image registration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    # Delete uploaded video
    video_path = Path(jobs[job_id].get("video_path", ""))
    if video_path.exists():
        video_path.unlink()

    # Delete uploaded images
    image_paths = jobs[job_id].get("image_paths", [])
    for img_path in image_paths:
        img_path = Path(img_path)
        if img_path.exists():
            img_path.unlink()

    # Remove job from storage
    del jobs[job_id]

    return {"message": "Job deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

