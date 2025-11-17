"""
PostgreSQL Manager
Handles all database operations for user management
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from loguru import logger
from pathlib import Path
from dotenv import load_dotenv
from .models import User, UserCreate, UserUpdate


class PostgresManager:
    """
    PostgreSQL database manager for user operations
    Handles connection pooling and CRUD operations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize PostgreSQL connection
        
        Args:
            config: Optional database configuration dict
                   If None, loads from configs/.env
        """
        # Load environment variables
        env_path = Path(__file__).parent.parent.parent / "configs" / ".env"
        if env_path.exists():
            load_dotenv(env_path)
        
        # Get configuration
        if config is None:
            config = {
                'host': os.getenv('POSTGRES_HOST', 'postgres_db'),
                'port': int(os.getenv('POSTGRES_PORT', 5432)),
                'user': os.getenv('POSTGRES_USER', 'hailt'),
                'password': os.getenv('POSTGRES_PASSWORD', '1'),
                'database': os.getenv('POSTGRES_DB', 'hailt_imespro')
            }
        
        self.config = config
        self.table_name = os.getenv('POSTGRES_TABLE', 'user')
        self.connection = None
        
        logger.info(f"PostgreSQL Manager initialized for {config['host']}:{config['port']}/{config['database']}")
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                dbname=self.config['database'],
                cursor_factory=RealDictCursor
            )
            logger.info(f"✅ Connected to PostgreSQL: {self.config['database']}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to PostgreSQL: {e}")
            return False
    
    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
            logger.info("Disconnected from PostgreSQL")
    
    def _ensure_connection(self):
        """Ensure database connection is active"""
        if self.connection is None or self.connection.closed:
            self.connect()
    
    def get_all_users(self) -> List[User]:
        """
        Get all users from database

        Returns:
            List of User objects
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f'SELECT * FROM "{self.table_name}" ORDER BY global_id')
                rows = cursor.fetchall()
                return [User(**dict(row)) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching users: {e}")
            return []
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """
        Get user by ID

        Args:
            user_id: User ID

        Returns:
            User object or None
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f'SELECT * FROM "{self.table_name}" WHERE id = %s', (user_id,))
                row = cursor.fetchone()
                return User(**dict(row)) if row else None
        except Exception as e:
            logger.error(f"Error fetching user {user_id}: {e}")
            return None
    
    def get_user_by_global_id(self, global_id: int) -> Optional[User]:
        """
        Get user by global_id

        Args:
            global_id: Global ID

        Returns:
            User object or None
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f'SELECT * FROM "{self.table_name}" WHERE global_id = %s', (global_id,))
                row = cursor.fetchone()
                return User(**dict(row)) if row else None
        except Exception as e:
            logger.error(f"Error fetching user with global_id {global_id}: {e}")
            return None

    def create_user(self, user_data: UserCreate) -> Optional[User]:
        """
        Create a new user

        Args:
            user_data: UserCreate object with user information

        Returns:
            Created User object or None
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO "{self.table_name}" (global_id, name, created_at, updated_at)
                    VALUES (%s, %s, NOW(), NOW())
                    RETURNING *
                    """,
                    (user_data.global_id, user_data.name)
                )
                row = cursor.fetchone()
                self.connection.commit()
                logger.info(f"✅ Created user: {user_data.name} (global_id: {user_data.global_id})")
                return User(**dict(row)) if row else None
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error creating user: {e}")
            return None

    def update_user(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """
        Update an existing user

        Args:
            user_id: User ID to update
            user_data: UserUpdate object with fields to update

        Returns:
            Updated User object or None
        """
        self._ensure_connection()
        try:
            # Build dynamic update query
            update_fields = []
            values = []

            if user_data.name is not None:
                update_fields.append("name = %s")
                values.append(user_data.name)

            if not update_fields:
                logger.warning("No fields to update")
                return self.get_user_by_id(user_id)

            update_fields.append("updated_at = NOW()")
            values.append(user_id)

            with self.connection.cursor() as cursor:
                query = f'UPDATE "{self.table_name}" SET {", ".join(update_fields)} WHERE id = %s RETURNING *'
                cursor.execute(query, values)
                row = cursor.fetchone()
                self.connection.commit()
                logger.info(f"✅ Updated user ID: {user_id}")
                return User(**dict(row)) if row else None
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error updating user {user_id}: {e}")
            return None

    def delete_user(self, user_id: int) -> bool:
        """
        Delete a user

        Args:
            user_id: User ID to delete

        Returns:
            True if successful, False otherwise
        """
        self._ensure_connection()
        try:
            with self.connection.cursor() as cursor:
                cursor.execute(f'DELETE FROM "{self.table_name}" WHERE id = %s', (user_id,))
                self.connection.commit()
                logger.info(f"✅ Deleted user ID: {user_id}")
                return True
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Error deleting user {user_id}: {e}")
            return False

    def get_users_dict(self) -> Dict[int, str]:
        """
        Get users as dictionary mapping global_id to name
        Useful for dropdowns and selectors

        Returns:
            Dict[global_id, name]
        """
        users = self.get_all_users()
        return {user.global_id: user.name for user in users}

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.disconnect()

