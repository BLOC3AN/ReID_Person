#!/usr/bin/env python3
"""
Database Service API
FastAPI service for PostgreSQL user management operations
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from loguru import logger

from services.database import PostgresManager, User
from services.database.models import UserCreate, UserUpdate

app = FastAPI(title="Database Service", version="1.0.0")

# Initialize database manager
db_manager = PostgresManager()


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup"""
    logger.info("🚀 Starting Database Service...")
    if db_manager.connect():
        logger.info("✅ Database Service ready")
    else:
        logger.error("❌ Failed to connect to database")


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown"""
    logger.info("Shutting down Database Service...")
    db_manager.disconnect()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "database"}


@app.get("/users", response_model=List[User])
async def get_all_users():
    """
    Get all users from database
    
    Returns:
        List of User objects
    """
    try:
        users = db_manager.get_all_users()
        return users
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/{user_id}", response_model=User)
async def get_user(user_id: int):
    """
    Get user by ID
    
    Args:
        user_id: User ID
        
    Returns:
        User object
    """
    try:
        user = db_manager.get_user_by_id(user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users/global/{global_id}", response_model=User)
async def get_user_by_global_id(global_id: int):
    """
    Get user by global_id
    
    Args:
        global_id: Global ID
        
    Returns:
        User object
    """
    try:
        user = db_manager.get_user_by_global_id(global_id)
        if user is None:
            raise HTTPException(status_code=404, detail="User not found")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user with global_id {global_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/users", response_model=User)
async def create_user(user_data: UserCreate):
    """
    Create a new user

    Args:
        user_data: UserCreate object

    Returns:
        Created User object
    """
    try:
        # Check if global_id already exists
        existing_user = db_manager.get_user_by_global_id(user_data.global_id)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail=f"User with global_id {user_data.global_id} already exists"
            )

        user = db_manager.create_user(user_data)
        if user is None:
            raise HTTPException(status_code=500, detail="Failed to create user")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/users/{user_id}", response_model=User)
async def update_user(user_id: int, user_data: UserUpdate):
    """
    Update an existing user

    Args:
        user_id: User ID to update
        user_data: UserUpdate object

    Returns:
        Updated User object
    """
    try:
        # Check if user exists
        existing_user = db_manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")

        user = db_manager.update_user(user_id, user_data)
        if user is None:
            raise HTTPException(status_code=500, detail="Failed to update user")
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/users/{user_id}")
async def delete_user(user_id: int):
    """
    Delete a user

    Args:
        user_id: User ID to delete

    Returns:
        Success message
    """
    try:
        # Check if user exists
        existing_user = db_manager.get_user_by_id(user_id)
        if not existing_user:
            raise HTTPException(status_code=404, detail="User not found")

        success = db_manager.delete_user(user_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete user")

        return JSONResponse(content={
            "message": f"User {user_id} deleted successfully"
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/users-dict")
async def get_users_dict():
    """
    Get users as dictionary mapping global_id to name
    Useful for dropdowns and selectors

    Returns:
        Dict[global_id, name]
    """
    try:
        users_dict = db_manager.get_users_dict()
        return users_dict
    except Exception as e:
        logger.error(f"Error fetching users dict: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)

