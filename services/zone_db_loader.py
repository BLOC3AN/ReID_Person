#!/usr/bin/env python3
"""
Zone Database Loader
Loads zone configurations from PostgreSQL database and converts to YAML format
Compatible with existing zone monitoring system
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from typing import Dict, List, Optional
from loguru import logger

from services.database import PostgresManager


def load_zones_from_db(camera_name: str = "camera_1", 
                       stream_url: Optional[str] = None) -> Dict:
    """
    Load zones from database and convert to zone config format
    
    Args:
        camera_name: Camera identifier (default: "camera_1")
        stream_url: Optional stream URL for reference
        
    Returns:
        Dict in zone config format compatible with ZoneMonitor
    """
    try:
        # Connect to database
        db_manager = PostgresManager()
        if not db_manager.connect():
            logger.error("Failed to connect to database")
            return None
        
        # Get all zones
        zones = db_manager.get_all_zones()
        
        if not zones:
            logger.warning("No zones found in database")
            return None
        
        # Convert to zone config format
        zone_config = {
            'cameras': {
                camera_name: {
                    'name': camera_name,
                    'zones': {}
                }
            }
        }
        
        if stream_url:
            zone_config['cameras'][camera_name]['stream_url'] = stream_url
        
        # Convert each zone
        for zone in zones:
            # Get users in this zone
            users = db_manager.get_users_by_zone(zone.zone_id)
            authorized_ids = [user.global_id for user in users]
            
            # Convert coordinates to polygon format
            # Database stores: x1, y1, x2, y2, x3, y3, x4, y4
            # Zone config expects: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            polygon = [
                [zone.x1, zone.y1],
                [zone.x2, zone.y2],
                [zone.x3, zone.y3],
                [zone.x4, zone.y4]
            ]
            
            # Add zone to config
            zone_config['cameras'][camera_name]['zones'][zone.zone_id] = {
                'name': zone.zone_name,
                'polygon': polygon,
                'authorized_ids': authorized_ids
            }
        
        logger.info(f"✅ Loaded {len(zones)} zones from database")
        db_manager.disconnect()
        
        return zone_config
        
    except Exception as e:
        logger.error(f"Error loading zones from database: {e}")
        return None


def save_zones_to_yaml(output_path: str, camera_name: str = "camera_1",
                       stream_url: Optional[str] = None) -> bool:
    """
    Load zones from database and save to YAML file
    
    Args:
        output_path: Path to save YAML file
        camera_name: Camera identifier
        stream_url: Optional stream URL
        
    Returns:
        True if successful, False otherwise
    """
    try:
        zone_config = load_zones_from_db(camera_name, stream_url)
        
        if not zone_config:
            logger.error("Failed to load zones from database")
            return False
        
        # Save to YAML
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(zone_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Saved zone config to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving zones to YAML: {e}")
        return False


def get_zone_config_dict(camera_name: str = "camera_1",
                         stream_url: Optional[str] = None) -> Optional[Dict]:
    """
    Get zone config as dictionary (for direct use without file)
    
    Args:
        camera_name: Camera identifier
        stream_url: Optional stream URL
        
    Returns:
        Zone config dictionary or None
    """
    return load_zones_from_db(camera_name, stream_url)


if __name__ == "__main__":
    # Test loading zones from database
    logger.info("Testing zone database loader...")
    
    # Load zones
    zone_config = load_zones_from_db("camera_1")
    
    if zone_config:
        logger.info("Zone config loaded successfully:")
        print(yaml.dump(zone_config, default_flow_style=False))
        
        # Save to file
        save_zones_to_yaml("configs/zones_from_db.yaml", "camera_1")
    else:
        logger.error("Failed to load zones from database")

