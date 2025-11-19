"""
Database Models
Defines data structures for database entities
"""

from typing import Optional
from pydantic import BaseModel
from datetime import datetime


class User(BaseModel):
    """User model matching PostgreSQL user table with Zone relationship (1:N)"""
    id: Optional[int] = None
    global_id: int
    name: str
    zone_id: Optional[str] = None  # Foreign key to working_zone.zone_id
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserCreate(BaseModel):
    """Model for creating a new user"""
    global_id: int
    name: str
    zone_id: Optional[str] = None  # Optional zone assignment


class UserUpdate(BaseModel):
    """Model for updating a user"""
    name: Optional[str] = None
    zone_id: Optional[str] = None  # Can update zone assignment


class WorkingZone(BaseModel):
    """WorkingZone model matching PostgreSQL working_zone table"""
    zone_id: str  # Primary key - Unique zone identifier (e.g., "ZONE_001")
    zone_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class WorkingZoneCreate(BaseModel):
    """Model for creating a new working zone"""
    zone_id: str
    zone_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float


class WorkingZoneUpdate(BaseModel):
    """Model for updating a working zone"""
    zone_name: Optional[str] = None
    x1: Optional[float] = None
    y1: Optional[float] = None
    x2: Optional[float] = None
    y2: Optional[float] = None
    x3: Optional[float] = None
    y3: Optional[float] = None
    x4: Optional[float] = None
    y4: Optional[float] = None


class WorkingZoneWithUsers(BaseModel):
    """WorkingZone with list of users (for 1:N relationship display)"""
    zone_id: str
    zone_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    x3: float
    y3: float
    x4: float
    y4: float
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    users: list = []  # List of User objects in this zone

    class Config:
        from_attributes = True

