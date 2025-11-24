"""
Database Models
Defines data structures for database entities
"""

from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class User(BaseModel):
    """User model matching PostgreSQL user table"""
    id: Optional[int] = None
    global_id: int
    name: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Model for creating a new user"""
    global_id: int
    name: str


class UserUpdate(BaseModel):
    """Model for updating a user"""
    name: Optional[str] = None

