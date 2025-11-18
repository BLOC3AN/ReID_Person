#!/usr/bin/env python3
"""
Working Zone Monitoring Script
Monitors person presence in defined working zones using IOU-based overlap detection
Uses R-tree spatial indexing for efficient zone lookup
"""

import sys
import os
import cv2
import csv
import yaml
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from rtree import index
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import (
    YOLOXDetector, ByteTrackWrapper, ArcFaceExtractor, QdrantVectorDB,
    ZoneMonitoringService, ZoneTask, ZoneResult, RedisTrackManager
)
from tabulate import tabulate


def calculate_iop(person_bbox, zone_bbox):
    """
    Calculate Intersection over Person (IoP)

    IoP = Intersection / Area_Person

    This measures what percentage of the person's bbox is inside the zone.
    - IoP = 1.0 means person is completely inside zone
    - IoP = 0.6 means 60% of person's body is inside zone
    - IoP = 0.0 means person is completely outside zone

    This is better than IOU for zone monitoring because:
    - Works correctly when zone is much larger than person
    - Intuitive: "60% of person in zone" = person is in zone
    - Independent of zone size

    Args:
        person_bbox: [x1, y1, x2, y2] or [x, y, w, h] format
        zone_bbox: [x1, y1, x2, y2] or [x, y, w, h] format
    Returns:
        IoP value (0-1): Percentage of person inside zone
    """
    # Auto-detect format: if x2 <= x1 or y2 <= y1, it's likely xywh
    def is_xywh_format(bbox):
        x1, y1, x2, y2 = bbox
        # If x2 <= x1 or y2 <= y1, definitely xywh
        if x2 <= x1 or y2 <= y1:
            return True
        return False

    # Convert person_bbox to xyxy
    if is_xywh_format(person_bbox):
        x1_p, y1_p, w_p, h_p = person_bbox
        x2_p, y2_p = x1_p + w_p, y1_p + h_p
    else:
        x1_p, y1_p, x2_p, y2_p = person_bbox

    # Convert zone_bbox to xyxy
    if is_xywh_format(zone_bbox):
        x1_z, y1_z, w_z, h_z = zone_bbox
        x2_z, y2_z = x1_z + w_z, y1_z + h_z
    else:
        x1_z, y1_z, x2_z, y2_z = zone_bbox

    # Calculate intersection
    x1_i = max(x1_p, x1_z)
    y1_i = max(y1_p, y1_z)
    x2_i = min(x2_p, x2_z)
    y2_i = min(y2_p, y2_z)

    # No intersection
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate person area
    area_person = (x2_p - x1_p) * (y2_p - y1_p)

    if area_person == 0:
        return 0.0

    # IoP = Intersection / Person Area
    iop = intersection / area_person

    return iop


class ZoneMonitor:
    """
    Monitor person presence in working zones using IoP-based overlap

    NEW LOGIC (Zone-Centric):
    - Each zone tracks which required persons are present/missing
    - Zone is COMPLETE when all required persons are present
    - Zone is INCOMPLETE when any required person is missing
    - Zones are independent - person can only be in one zone at a time
    """

    def __init__(self, zone_config_path, iou_threshold=0.6, zone_opacity=0.3, num_cameras=1):
        """
        Args:
            zone_config_path: Path to zone configuration YAML
            iou_threshold: IoP threshold for zone overlap (default: 0.6 = 60% of person in zone)
                          Note: Parameter name kept as 'iou_threshold' for backward compatibility,
                          but it's actually used as IoP (Intersection over Person) threshold
            zone_opacity: Zone border thickness factor (default: 0.3, range: 0.0-1.0)
                         Converts to pixel thickness: 0.0-1.0 → 1-10 pixels
            num_cameras: Number of cameras (default: 1 for single camera)
        """
        self.num_cameras = num_cameras
        self.zones, self.is_multi_camera = self._load_zones(zone_config_path)
        self.iop_threshold = iou_threshold  # Actually IoP threshold
        self.zone_opacity = zone_opacity  # Actually border thickness factor
        self.rtree_idx = self._build_rtree()

        # NEW: Zone-centric state tracking
        self.zone_status = self._initialize_zone_status()  # Track each zone's completeness
        self.person_locations = {}  # Track where each person is currently located
        self.zone_violations = []   # List of zone violations (zone incomplete events)
        self.last_violation_table_print = 0

        if self.is_multi_camera:
            logger.info(f"✅ ZoneMonitor initialized for {self.num_cameras} cameras with {len(self.zones)} total zones")
            for camera_idx in range(self.num_cameras):
                camera_zones = [z for z in self.zones.values() if z.get('camera_idx') == camera_idx]
                logger.info(f"   Camera {camera_idx+1}: {len(camera_zones)} zones")
        else:
            logger.info(f"✅ ZoneMonitor initialized with {len(self.zones)} zones")
        logger.info(f"   IoP threshold: {iou_threshold*100:.0f}% (percentage of person in zone)")
        thickness_px = max(1, int(zone_opacity * 10)) if zone_opacity > 0 else 3
        logger.info(f"   Zone border thickness: {thickness_px}px")
        logger.info(f"   📊 Logic: Zone-centric (each zone checks for required persons)")

    def _load_zones(self, config_path):
        """
        Load zone definitions from YAML or JSON

        Supports two formats:
        1. Single camera (old format):
           zones:
             zone1: {name, polygon, authorized_ids}
             zone2: {...}

        2. Multi-camera (new format):
           cameras:
             camera_1:
               zones:
                 zone1: {name, polygon, authorized_ids}
             camera_2:
               zones:
                 zone1: {...}

        Returns:
            (zones_dict, is_multi_camera)
        """
        # Load config file (support both YAML and JSON)
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                config = yaml.safe_load(f)

        zones = {}
        is_multi_camera = False

        # Check format: 'cameras' key = multi-camera, 'zones' key = single camera
        if 'cameras' in config:
            # Multi-camera format
            is_multi_camera = True
            logger.info("📹 Detected multi-camera zone config")

            for camera_idx, (camera_id, camera_data) in enumerate(config['cameras'].items()):
                camera_name = camera_data.get('name', f'Camera {camera_idx+1}')
                logger.info(f"   Loading zones for {camera_name}")

                if 'zones' not in camera_data:
                    logger.warning(f"   No zones defined for {camera_name}")
                    continue

                for zone_id, zone_data in camera_data['zones'].items():
                    # Create unique zone ID: camera_id + zone_id
                    unique_zone_id = f"{camera_id}_{zone_id}"

                    # Create zone with camera metadata
                    zone = self._create_zone_from_data(
                        zone_data,
                        camera_idx=camera_idx,
                        camera_id=camera_id,
                        camera_name=camera_name
                    )
                    zones[unique_zone_id] = zone
                    logger.info(f"      - {zone_data['name']} ({len(zone_data.get('authorized_ids', []))} authorized)")

        elif 'zones' in config:
            # Single camera format (backward compatible)
            logger.info("📹 Detected single-camera zone config")

            for zone_id, zone_data in config['zones'].items():
                # Create zone for camera 0 (single camera)
                zone = self._create_zone_from_data(zone_data, camera_idx=0)
                zones[zone_id] = zone
        else:
            raise ValueError("Invalid zone config: must contain 'zones' or 'cameras' key")

        return zones, is_multi_camera

    def _create_zone_from_data(self, zone_data, camera_idx=0, camera_id=None, camera_name=None):
        """
        Create zone dict from zone data (DRY helper).

        Args:
            zone_data: Dict with 'name', 'polygon', 'authorized_ids'
            camera_idx: Camera index (0-based)
            camera_id: Camera ID string (optional, for multi-camera)
            camera_name: Camera name (optional, for multi-camera)

        Returns:
            Zone dict with bbox, polygon, and metadata
        """
        # Validate required fields
        if 'name' not in zone_data:
            raise ValueError(f"Zone data missing required field 'name': {zone_data}")

        if 'polygon' not in zone_data:
            raise ValueError(f"Zone '{zone_data.get('name', 'Unknown')}' missing required field 'polygon'. "
                           f"Make sure your zone config file contains polygon coordinates for each zone.")

        # Convert polygon to bbox [x1, y1, x2, y2]
        polygon = np.array(zone_data['polygon'])

        # Validate polygon has at least 3 points
        if len(polygon) < 3:
            raise ValueError(f"Zone '{zone_data['name']}' polygon must have at least 3 points, got {len(polygon)}")

        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)

        zone = {
            'name': zone_data['name'],
            'bbox': [x1, y1, x2, y2],
            'polygon': polygon,
            'authorized_ids': zone_data.get('authorized_ids', []),
            'camera_idx': camera_idx
        }

        # Add camera metadata if provided (multi-camera mode)
        if camera_id is not None:
            zone['camera_id'] = camera_id
        if camera_name is not None:
            zone['camera_name'] = camera_name

        return zone

    def _initialize_zone_status(self):
        """
        Initialize zone status tracking (ZONE-CENTRIC LOGIC)

        Each zone tracks:
        - required_persons: List of person IDs that should be in this zone (authorized_ids)
        - present_persons: Dict of persons currently in this zone {person_id: {name, enter_time, duration}}
        - missing_persons: List of person IDs missing from this zone
        - is_complete: Boolean - True if all required persons are present
        - violation_start_time: Timestamp when zone became incomplete (None if complete)

        Returns:
            Dict[zone_id, zone_state]
        """
        status = {}
        for zone_id, zone_data in self.zones.items():
            required = zone_data.get('authorized_ids', [])
            status[zone_id] = {
                'name': zone_data['name'],
                'required_persons': required,
                'present_persons': {},  # Will be populated as persons are detected
                'missing_persons': required.copy(),  # Initially all are missing
                'is_complete': False,  # All zones start as incomplete (even empty zones)
                'violation_start_time': None,
                'camera_idx': zone_data.get('camera_idx', 0)
            }
        return status

    def _build_rtree(self):
        """Build R-tree spatial index for zones"""
        idx = index.Index()
        for i, (zone_id, zone_data) in enumerate(self.zones.items()):
            bbox = zone_data['bbox']
            # R-tree format: (minx, miny, maxx, maxy)
            idx.insert(i, tuple(bbox), obj=zone_id)
        
        logger.info(f"   R-tree index built for {len(self.zones)} zones")
        return idx
    
    def find_zone(self, person_bbox, camera_idx=0):
        """
        Find which zone contains the person using IoP >= threshold

        IoP (Intersection over Person) measures what percentage of the person
        is inside the zone. This works correctly even when zone is much larger
        than person bbox.

        Args:
            person_bbox: [x1, y1, x2, y2] or [x, y, w, h]
            camera_idx: Camera index (0-based) for multi-camera setup
        Returns:
            zone_id or None
        """
        # Convert to x1,y1,x2,y2 format
        if len(person_bbox) == 4:
            x, y, w, h = person_bbox
            person_bbox_xyxy = [x, y, x+w, y+h]
        else:
            person_bbox_xyxy = person_bbox

        # R-tree candidates (fast spatial query)
        candidates = list(self.rtree_idx.intersection(person_bbox_xyxy, objects=True))

        # Find best IoP match (only check zones for this camera)
        best_zone = None
        best_iop = 0.0

        for candidate in candidates:
            zone_id = candidate.object
            zone_data = self.zones[zone_id]

            # Skip zones from other cameras
            if zone_data.get('camera_idx', 0) != camera_idx:
                continue

            zone_bbox = zone_data['bbox']

            # Calculate IoP (Intersection over Person)
            iop = calculate_iop(person_bbox_xyxy, zone_bbox)

            if iop >= self.iop_threshold and iop > best_iop:
                best_iop = iop
                best_zone = zone_id

        return best_zone
    
    def update_presence(self, global_id, zone_id, frame_time, person_name):
        """
        Update person location and recalculate zone status (ZONE-CENTRIC LOGIC)

        NEW LOGIC:
        1. Update person's current location
        2. Recalculate affected zones to check completeness
        3. Generate violations for incomplete zones

        Args:
            global_id: Person's global ID
            zone_id: Current zone ID (or None if outside all zones)
            frame_time: Current timestamp (seconds)
            person_name: Person's name
        """
        # Get old location
        old_zone = self.person_locations.get(global_id, {}).get('current_zone')

        # Check if person moved
        if old_zone != zone_id:
            # Update person location
            self.person_locations[global_id] = {
                'name': person_name,
                'current_zone': zone_id,
                'enter_time': frame_time if zone_id else None,
                'camera_idx': 0  # Will be updated by caller if needed
            }

            # Log movement
            if old_zone and zone_id:
                logger.info(f"🚶 {person_name} (ID:{global_id}) moved: "
                           f"{self.zones[old_zone]['name']} → {self.zones[zone_id]['name']}")
            elif zone_id:
                logger.info(f"➡️  {person_name} (ID:{global_id}) entered "
                           f"{self.zones[zone_id]['name']}")
            elif old_zone:
                logger.info(f"⬅️  {person_name} (ID:{global_id}) left "
                           f"{self.zones[old_zone]['name']}")

            # Recalculate affected zones
            affected_zones = set()
            if old_zone:
                affected_zones.add(old_zone)
            if zone_id:
                affected_zones.add(zone_id)

            # Also check all zones that require this person
            for zid, zone_data in self.zones.items():
                if global_id in zone_data.get('authorized_ids', []):
                    affected_zones.add(zid)

            # Update all affected zones
            for affected_zone_id in affected_zones:
                self._update_zone_status(affected_zone_id, frame_time)
        else:
            # Person still in same zone - just update duration
            if zone_id and global_id in self.person_locations:
                enter_time = self.person_locations[global_id].get('enter_time')
                if enter_time:
                    # Update duration in zone_status
                    zone_state = self.zone_status.get(zone_id)
                    if zone_state and global_id in zone_state['present_persons']:
                        zone_state['present_persons'][global_id]['duration'] = frame_time - enter_time

    def _update_zone_status(self, zone_id, frame_time):
        """
        Update status for a specific zone (ZONE-CENTRIC LOGIC)

        Check which required persons are present in this zone:
        - Update present_persons dict
        - Update missing_persons list
        - Update is_complete flag
        - Track violation timing

        Args:
            zone_id: Zone to update
            frame_time: Current timestamp
        """
        zone_state = self.zone_status[zone_id]
        required = zone_state['required_persons']
        present = {}
        missing = []

        # Check each required person
        for person_id in required:
            if person_id in self.person_locations:
                person_loc = self.person_locations[person_id]
                if person_loc['current_zone'] == zone_id:
                    # Person is in this zone
                    enter_time = person_loc.get('enter_time', frame_time)
                    present[person_id] = {
                        'name': person_loc['name'],
                        'enter_time': enter_time,
                        'duration': frame_time - enter_time
                    }
                else:
                    # Person is elsewhere
                    missing.append(person_id)
            else:
                # Person not detected yet
                missing.append(person_id)

        # Check if missing persons changed
        old_missing = zone_state.get('missing_persons', [])
        missing_changed = set(old_missing) != set(missing)

        # Update zone state
        zone_state['present_persons'] = present
        zone_state['missing_persons'] = missing
        was_complete = zone_state['is_complete']
        # Zone is complete only if:
        # 1. Has required persons (not empty) - empty zones are always incomplete
        # 2. All required persons are present (no missing)
        zone_state['is_complete'] = (len(required) > 0 and len(missing) == 0)

        # Track violation timing and log state changes
        if not zone_state['is_complete']:
            if zone_state['violation_start_time'] is None:
                # Zone just became incomplete - LOG
                zone_state['violation_start_time'] = frame_time
                self._log_zone_violation(zone_id, missing, frame_time)
            elif missing_changed:
                # Zone still incomplete but missing persons changed - LOG
                self._log_zone_violation(zone_id, missing, frame_time)
        else:
            # Zone is complete
            if not was_complete and zone_state['violation_start_time'] is not None:
                # Zone just became complete - LOG resolution
                duration = frame_time - zone_state['violation_start_time']
                logger.info(f"✅ Zone '{zone_state['name']}' now complete (was incomplete for {duration:.1f}s)")
            zone_state['violation_start_time'] = None

    def _log_zone_violation(self, zone_id, missing_person_ids, frame_time):
        """Log zone violation when zone becomes incomplete"""
        zone_state = self.zone_status[zone_id]
        missing_names = []
        for pid in missing_person_ids:
            if pid in self.person_locations:
                missing_names.append(self.person_locations[pid]['name'])
            else:
                missing_names.append(f"Person {pid}")

        # Add to violations list
        violation = {
            'zone_id': zone_id,
            'zone_name': zone_state['name'],
            'missing_persons': missing_person_ids,
            'missing_names': missing_names,
            'time': frame_time,
            'type': 'zone_incomplete'
        }
        self.zone_violations.append(violation)

        # Log to console
        missing_str = ", ".join([f"{name} (ID:{pid})"
                                for pid, name in zip(missing_person_ids, missing_names)])
        logger.warning(f"🚨 Zone '{zone_state['name']}' incomplete: Missing {missing_str}")

    def _log_zone_event(self, event_type, global_id, zone_id, time, duration):
        """
        Log zone entry/exit events (DEPRECATED - kept for backward compatibility)

        Note: This method is no longer used in zone-centric logic but kept
        in case external code calls it.
        """
        pass  # No-op in new logic

    def print_violation_table(self, current_time, alert_threshold=0):
        """
        Print zone status table (ZONE-CENTRIC LOGIC)

        Shows status of each zone:
        - Which persons are required
        - Which persons are present
        - Which persons are missing
        - Zone completeness status

        Args:
            current_time: Current frame time in seconds
            alert_threshold: Time threshold (seconds) before showing alert
        """
        table_data = []
        headers = ["Zone", "Required", "Present", "Missing", "Status", "Duration (s)"]

        for zone_id, zone_state in self.zone_status.items():
            # Build required persons string
            required_str = ", ".join([str(pid) for pid in zone_state['required_persons']])
            if not required_str:
                required_str = "None"

            # Build present persons string
            present_ids = list(zone_state['present_persons'].keys())
            present_str = ", ".join([str(pid) for pid in present_ids])
            if not present_str:
                present_str = "None"

            # Build missing persons string
            missing_str = ", ".join([str(pid) for pid in zone_state['missing_persons']])
            if not missing_str:
                missing_str = "None"

            # Determine status
            if zone_state['is_complete']:
                status = "✅ Complete"
                duration = 0
            else:
                violation_start = zone_state['violation_start_time']
                if violation_start is not None:
                    duration = current_time - violation_start
                    if duration >= alert_threshold:
                        status = "🚨 ALERT"
                    else:
                        status = "⚠️  Incomplete"
                else:
                    status = "⚠️  Incomplete"
                    duration = 0

            table_data.append([
                zone_state['name'],
                required_str,
                present_str,
                missing_str,
                status,
                f"{duration:.1f}"
            ])

        if table_data:
            logger.info("\n" + "="*80)
            logger.info("📊 ZONE STATUS MONITORING")
            logger.info("="*80)
            logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
            logger.info("="*80 + "\n")
    
    def get_zone_summary(self):
        """
        Get summary of zone status (ZONE-CENTRIC LOGIC)

        Returns:
            Dict with zone status information
        """
        summary = {}
        for zone_id, zone_state in self.zone_status.items():
            # Build list of present persons with details
            present_list = [
                {
                    'id': pid,
                    'name': pdata['name'],
                    'duration': pdata['duration']
                }
                for pid, pdata in zone_state['present_persons'].items()
            ]

            summary[zone_id] = {
                'name': zone_state['name'],
                'required_persons': zone_state['required_persons'],
                'present_persons': present_list,
                'missing_persons': zone_state['missing_persons'],
                'is_complete': zone_state['is_complete'],
                'status': '✅ Complete' if zone_state['is_complete'] else '🚫 Incomplete',
                'count': len(present_list)
            }
        return summary

    def get_violations(self):
        """
        Get all zone violations (ZONE-CENTRIC LOGIC)

        Returns:
            List of zone violation events
        """
        return self.zone_violations.copy()
    
    def save_report(self, output_path):
        """Save zone monitoring report to JSON (ZONE-CENTRIC LOGIC)"""
        report = {
            'summary': self.get_zone_summary(),
            'violations': self.get_violations(),
            'zones': {
                zone_id: {
                    'name': data['name'],
                    'required_persons': data.get('authorized_ids', []),
                    'camera_idx': data.get('camera_idx', 0)
                }
                for zone_id, data in self.zones.items()
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📄 Zone report saved to: {output_path}")

    def draw_ruler(self, frame):
        """Draw ruler on frame edges for coordinate reference"""
        h, w = frame.shape[:2]

        # Ruler settings
        ruler_color = (200, 200, 200)  # Light gray
        text_color = (255, 255, 255)   # White
        major_tick = 100  # Major tick every 100 pixels
        minor_tick = 50   # Minor tick every 50 pixels

        # Top ruler (horizontal)
        for x in range(0, w, minor_tick):
            if x % major_tick == 0:
                # Major tick
                cv2.line(frame, (x, 0), (x, 15), ruler_color, 1)
                if x > 0:  # Don't draw 0 at corner
                    cv2.putText(frame, str(x), (x-15, 12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
            else:
                # Minor tick
                cv2.line(frame, (x, 0), (x, 8), ruler_color, 1)

        # Bottom ruler (horizontal)
        for x in range(0, w, minor_tick):
            if x % major_tick == 0:
                # Major tick
                cv2.line(frame, (x, h-15), (x, h), ruler_color, 1)
                if x > 0:
                    cv2.putText(frame, str(x), (x-15, h-3),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
            else:
                # Minor tick
                cv2.line(frame, (x, h-8), (x, h), ruler_color, 1)

        # Left ruler (vertical)
        for y in range(0, h, minor_tick):
            if y % major_tick == 0:
                # Major tick
                cv2.line(frame, (0, y), (15, y), ruler_color, 1)
                if y > 0:
                    cv2.putText(frame, str(y), (2, y+4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
            else:
                # Minor tick
                cv2.line(frame, (0, y), (8, y), ruler_color, 1)

        # Right ruler (vertical)
        for y in range(0, h, minor_tick):
            if y % major_tick == 0:
                # Major tick
                cv2.line(frame, (w-15, y), (w, y), ruler_color, 1)
                if y > 0:
                    cv2.putText(frame, str(y), (w-35, y+4),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, text_color, 1)
            else:
                # Minor tick
                cv2.line(frame, (w-8, y), (w, y), ruler_color, 1)

        # Draw corner coordinates
        cv2.putText(frame, "0,0", (5, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(frame, f"{w},{h}", (w-60, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return frame

    def draw_zones(self, frame, camera_idx=0):
        """
        Draw zone boundaries on frame (ZONE-CENTRIC LOGIC)

        Zone border color logic:
        - Green (0, 255, 0): Zone is COMPLETE (all required persons present)
        - Red (0, 0, 255): Zone is INCOMPLETE (missing required persons)

        Args:
            frame: Frame to draw on
            camera_idx: Camera index (0-based) for multi-camera setup
        """
        # Draw ruler first (background layer)
        frame = self.draw_ruler(frame)

        # Draw zones with status-based colors (only for this camera)
        for zone_id, zone_data in self.zones.items():
            # Skip zones from other cameras
            if zone_data.get('camera_idx', 0) != camera_idx:
                continue

            polygon = zone_data['polygon']
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)

            # Determine zone color based on completeness
            zone_state = self.zone_status[zone_id]
            if zone_state['is_complete']:
                color = (0, 255, 0)  # Green - zone complete
            else:
                color = (0, 0, 255)  # Red - zone incomplete

            # Draw polygon border only (no background fill)
            # Use zone_opacity to control line thickness (convert 0.0-1.0 to 1-10 pixels)
            thickness = max(1, int(self.zone_opacity * 10)) if self.zone_opacity > 0 else 3
            cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)

            # Draw zone name with background (same color as border)
            x, y = polygon[0]
            zone_name = zone_data['name']

            # Add camera info for multi-camera
            if self.is_multi_camera:
                zone_name = f"[Cam{camera_idx+1}] {zone_name}"

            # Add status indicator
            status_icon = "✅" if zone_state['is_complete'] else "🚫"
            zone_name = f"{status_icon} {zone_name}"

            text_size = cv2.getTextSize(zone_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]

            # Background rectangle for text (same color as border)
            cv2.rectangle(frame,
                         (int(x), int(y)-text_size[1]-15),
                         (int(x)+text_size[0]+10, int(y)-5),
                         color, -1)

            # Text in white
            cv2.putText(frame, zone_name, (int(x)+5, int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame


def _process_reid_logic_zone(track_id, frame_id, current_frame_count, embedding,
                             database, similarity_threshold, redis_manager,
                             track_labels, logger):
    """
    Process ReID logic with Redis storage for zone monitoring

    Decision Logic:
    - Frame 1: Extract → Assign (first time)
    - Frame 60+: Extract → Compare with old
    - Update only if new_sim > old_sim AND passes threshold
    - Reset when track lost (TTL expires)

    Returns:
        Updated track_data dict or None
    """
    matches = database.find_best_match(embedding, threshold=0.0, top_k=1)

    if not matches:
        return None

    new_global_id, new_similarity, new_person_name = matches[0]

    # Get old data from Redis or in-memory
    old_data = None
    if redis_manager:
        old_data = redis_manager.get_track(track_id)
    if old_data is None:
        old_data = track_labels.get(track_id)

    # Decision logic
    if new_similarity >= similarity_threshold:
        if old_data is None:
            # New track (first time or recovered after TTL)
            new_data = {
                'global_id': new_global_id,
                'similarity': new_similarity,
                'best_similarity': new_similarity,
                'person_name': new_person_name,
                'label': new_person_name,
                'first_assignment_frame': frame_id,
                'last_update_frame': frame_id,
                'timestamp': time.time(),
                'camera_idx': 0,
                'status': 'active'
            }
            logger.info(f"✅ Track {track_id}: ASSIGN {new_person_name} (ID:{new_global_id}, sim={new_similarity:.4f})")

        else:
            # Existing track
            old_similarity = old_data.get('similarity', 0.0)
            old_person_name = old_data.get('person_name', 'Unknown')

            if new_similarity > old_similarity:
                # UPDATE: Better match found
                new_data = old_data.copy()
                new_data['global_id'] = new_global_id
                new_data['similarity'] = new_similarity
                new_data['best_similarity'] = max(new_similarity, old_data.get('best_similarity', 0.0))
                new_data['person_name'] = new_person_name
                new_data['label'] = new_person_name
                new_data['last_update_frame'] = frame_id
                new_data['timestamp'] = time.time()
                logger.info(f"✅ Track {track_id}: UPDATE {old_person_name} → {new_person_name} (sim {old_similarity:.4f} → {new_similarity:.4f})")

            else:
                # REJECT: Same or worse match
                new_data = old_data.copy()
                new_data['last_update_frame'] = frame_id
                new_data['timestamp'] = time.time()
                logger.debug(f"❌ Track {track_id}: REJECT {new_person_name} (sim {new_similarity:.4f} < {old_similarity:.4f})")

    else:
        # FAIL_THRESHOLD: Below threshold
        if old_data is None:
            logger.debug(f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (sim {new_similarity:.4f} < {similarity_threshold:.4f})")
            return None
        else:
            new_data = old_data.copy()
            new_data['last_update_frame'] = frame_id
            new_data['timestamp'] = time.time()
            logger.debug(f"❌ Track {track_id}: FAIL_THRESHOLD {new_person_name} (sim {new_similarity:.4f} < {similarity_threshold:.4f})")

    # Save to Redis and in-memory
    if redis_manager:
        redis_manager.set_track(track_id, new_data)
    track_labels[track_id] = new_data

    return new_data


def process_video_with_zones(video_path, zone_config_path, reid_config_path=None,
                             similarity_threshold=0.8, iou_threshold=0.6, zone_opacity=0.3,
                             output_dir=None, max_frames=None, max_duration_seconds=None,
                             output_video_path=None, output_csv_path=None, output_json_path=None,
                             progress_callback=None, cancellation_flag=None,
                             violation_callback=None, alert_threshold=0):
    """
    Process video with zone monitoring integrated into ReID pipeline

    Args:
        video_path: Path to input video or stream URL(s)
        zone_config_path: Path to zone configuration YAML
        reid_config_path: Path to ReID config (default: configs/config.yaml)
        similarity_threshold: ReID similarity threshold
        iou_threshold: Zone IoP threshold (default: 0.6 = 60% of person in zone)
                      Note: Parameter name is 'iou_threshold' for backward compatibility,
                      but it's actually IoP (Intersection over Person) threshold.
                      IoP = 0.6 means 60% of person's body is inside the zone.
        zone_opacity: Zone border thickness factor (default: 0.3, range: 0.0-1.0)
                     Converts to pixel thickness: 0.0-1.0 → 1-10 pixels
        output_dir: Output directory for results (used if specific paths not provided)
        max_frames: Maximum frames to process
        max_duration_seconds: Maximum duration in seconds to process (converted to frames)
        output_video_path: Specific path for output video (optional)
        output_csv_path: Specific path for output CSV (optional)
        output_json_path: Specific path for output JSON (optional)
        progress_callback: Optional callback function(frame_id, tracks) for progress updates
        cancellation_flag: Optional threading.Event() to signal cancellation
        violation_callback: Optional callback(violation_dict) called when violation occurs
        alert_threshold: Time threshold (seconds) before showing alert (default: 0)
    """
    from detect_and_track import PersonReIDPipeline
    from utils.multi_stream_reader import MultiStreamReader, parse_stream_urls

    logger.info("="*80)
    logger.info("🎯 ZONE MONITORING WITH PERSON REID")
    logger.info("="*80)

    # Initialize ReID pipeline
    pipeline = PersonReIDPipeline(reid_config_path)
    pipeline.initialize_detector()
    pipeline.initialize_tracker()
    pipeline.initialize_extractor()
    pipeline.initialize_database()

    # Parse video_path to check if it contains multiple URLs
    urls = parse_stream_urls(video_path)
    num_cameras = len(urls)

    # Initialize zone monitor with camera count
    zone_monitor = ZoneMonitor(zone_config_path, iou_threshold, zone_opacity, num_cameras=num_cameras)

    # Initialize zone monitoring service (separate thread)
    zone_service = ZoneMonitoringService(zone_monitor, max_queue_size=100)
    zone_service.start()

    # Initialize Redis Track Manager
    redis_host = os.getenv('REDIS_HOST', 'localhost')
    redis_port = int(os.getenv('REDIS_PORT', 6379))
    redis_ttl = int(os.getenv('REDIS_TTL', 300))

    try:
        redis_manager = RedisTrackManager(host=redis_host, port=redis_port, ttl=redis_ttl)
        logger.info(f"✅ Redis Track Manager initialized")
    except Exception as e:
        logger.warning(f"⚠️  Redis not available: {e}. Using in-memory storage only.")
        redis_manager = None

    # Setup output paths
    if output_video_path and output_csv_path and output_json_path:
        # Use provided paths
        output_video = Path(output_video_path)
        output_csv = Path(output_csv_path)
        output_json = Path(output_json_path)
    else:
        # Generate paths with timestamp
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"
        else:
            output_dir = Path(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem

        output_video = output_dir / "videos" / f"{video_name}_zones_{timestamp}.mp4"
        output_csv = output_dir / "csv" / f"{video_name}_zones_{timestamp}.csv"
        output_json = output_dir / "logs" / f"{video_name}_zones_{timestamp}.json"

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Create extracted_objects directory for person frames
    # Use outputs/extracted_objects/ which is mounted in Docker
    extracted_objects_dir = Path(__file__).parent.parent / "outputs" / "extracted_objects"
    extracted_objects_dir.mkdir(parents=True, exist_ok=True)

    # Track frame folders for each person (changed from video writers to frame folders)
    person_video_writers = {}  # {track_id: {'folder': Path, 'label': str, 'path': Path, 'frame_count': int}}

    # Open video/stream using appropriate reader
    from utils.stream_reader import StreamReader

    try:
        if num_cameras > 1:
            # Multiple cameras - use MultiStreamReader
            logger.info(f"Opening {num_cameras} camera streams")
            stream_reader = MultiStreamReader(urls, use_ffmpeg_for_udp=True)
        else:
            # Single camera - use StreamReader
            stream_reader = StreamReader(video_path, use_ffmpeg_for_udp=True)

        props = stream_reader.get_properties()
        fps = props['fps']
        width = props['width']
        height = props['height']
        total_frames = props.get('total_frames', 0)
        is_stream = props['is_stream']

        logger.info(f"Video: {video_path}")
        logger.info(f"Resolution: {width}x{height} @ {fps:.1f} FPS")
        if not is_stream:
            logger.info(f"Total frames: {total_frames}")
        else:
            logger.info(f"Stream mode (unlimited frames)")

        # Calculate max frames from duration if specified
        if max_duration_seconds is not None and max_frames is None:
            max_frames = int(max_duration_seconds * fps)
            logger.info(f"Max duration: {max_duration_seconds}s → {max_frames} frames at {fps:.1f} FPS")

    except Exception as e:
        logger.error(f"Failed to open video/stream: {e}")
        return

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(str(output_video), fourcc, int(fps), (width, height))

    # CSV writer (ZONE-CENTRIC LOGIC)
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_id', 'track_id', 'global_id', 'person_name', 'similarity',
        'x', 'y', 'w', 'h', 'zone_id', 'zone_name', 'duration_in_zone', 'camera_idx'
    ])

    # Processing state
    frame_id = 0
    track_labels = {}
    track_frame_count = {}

    # FPS tracking
    fps_history = []
    fps_window_size = 30  # Average over last 30 frames
    last_frame_time = None

    logger.info("="*80)
    logger.info("🚀 Starting Processing...")
    logger.info("="*80)

    import time
    start_time = time.time()

    while True:
        # Check cancellation flag
        if cancellation_flag is not None and cancellation_flag.is_set():
            logger.info("Processing cancelled by user")
            break

        ret, frame = stream_reader.read()
        if not ret or frame is None:
            break

        if max_frames and frame_id >= max_frames:
            logger.info(f"Reached max frames limit: {max_frames}")
            break

        frame_time = frame_id / fps  # Time in seconds

        # Calculate FPS
        current_time = time.time()
        if last_frame_time is not None:
            frame_fps = 1.0 / (current_time - last_frame_time)
            fps_history.append(frame_fps)
            if len(fps_history) > fps_window_size:
                fps_history.pop(0)
        last_frame_time = current_time

        avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

        # Get camera metadata if multi-stream
        camera_metadata = getattr(stream_reader, 'camera_metadata', None)

        # Detect
        detections = pipeline.detector.detect(frame)

        # Track
        tracks = pipeline.tracker.update(detections, (height, width))

        if frame_id % 30 == 0:
            logger.info(f"Frame {frame_id}/{total_frames}: {len(tracks)} tracks")

        # Initialize zone_ids for this frame (will be filled during track processing)
        zone_ids_for_service = {}

        # Process each track
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            track_id = int(track_id)

            # Convert to xywh
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

            # Initialize track
            if track_id not in track_frame_count:
                track_frame_count[track_id] = 0

            track_frame_count[track_id] += 1
            current_frame_count = track_frame_count[track_id]

            # ReID extraction: Frame 1 + re-verify every 60 frames
            should_extract = (current_frame_count == 1) or (current_frame_count % 60 == 0)

            if should_extract:
                bbox = [x, y, w, h]
                embedding = pipeline.extractor.extract(frame, bbox)
                _process_reid_logic_zone(
                    track_id=track_id,
                    frame_id=frame_id,
                    current_frame_count=current_frame_count,
                    embedding=embedding,
                    database=pipeline.database,
                    similarity_threshold=similarity_threshold,
                    redis_manager=redis_manager,
                    track_labels=track_labels,
                    logger=logger
                )

            # Get track info
            info = track_labels.get(track_id, {
                'global_id': -1,
                'similarity': 0.0,
                'label': 'Unknown',
                'person_name': 'Unknown'
            })

            # ZONE MONITORING: Check which zone person is in (Main thread only)
            person_bbox = [x, y, w, h]

            # Convert bbox to camera-relative coordinates (single camera = no conversion)
            if num_cameras > 1:
                relative_bbox, camera_idx = stream_reader.bbox_to_camera_relative(person_bbox)
            else:
                relative_bbox, camera_idx = person_bbox, 0

            # Find zone (works for both single and multi-camera)
            # This is done in main thread for immediate use in drawing
            zone_id = zone_monitor.find_zone(relative_bbox, camera_idx)

            # Store zone_id for zone service
            zone_ids_for_service[track_id] = zone_id

            # Get zone info (ZONE-CENTRIC LOGIC)
            zone_name = zone_monitor.zones[zone_id]['name'] if zone_id else "None"
            duration = 0.0

            # Get person location data for duration tracking (from cached results)
            if info['global_id'] > 0 and info['global_id'] in zone_monitor.person_locations:
                person_loc = zone_monitor.person_locations[info['global_id']]
                if zone_id and person_loc.get('enter_time'):
                    duration = frame_time - person_loc['enter_time']

            # Write to CSV (ZONE-CENTRIC LOGIC)
            csv_writer.writerow([
                frame_id, track_id, info['global_id'], info['person_name'],
                f"{info['similarity']:.4f}", x, y, w, h,
                zone_id if zone_id else "", zone_name, f"{duration:.2f}", camera_idx
            ])

            # Save person frames as images
            # Create folder for this track if not exists
            if track_id not in person_video_writers:
                # Get label for folder name
                person_label = info['label']

                # Create folder for this person
                person_folder = extracted_objects_dir / person_label
                person_folder.mkdir(parents=True, exist_ok=True)

                # Generate unique folder name with timestamp
                folder_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
                track_folder_name = f"{person_label}_{folder_timestamp}_track{track_id}"
                track_folder_path = person_folder / track_folder_name
                track_folder_path.mkdir(parents=True, exist_ok=True)

                person_video_writers[track_id] = {
                    'folder': track_folder_path,
                    'label': person_label,
                    'path': track_folder_path,
                    'frame_count': 0
                }

                logger.info(f"  📁 Created frames folder for {person_label} (Track {track_id}): {track_folder_name}")

            # Save cropped person frame to folder
            if track_id in person_video_writers:
                try:
                    # Validate bounding box first
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        logger.warning(f"⚠️ Invalid bbox for track {track_id} at frame {frame_id}: ({x},{y},{w},{h})")
                        continue

                    # Clip to frame boundaries
                    frame_h, frame_w = frame.shape[:2]
                    x_clipped = max(0, x)
                    y_clipped = max(0, y)
                    w_clipped = min(w, frame_w - x_clipped)
                    h_clipped = min(h, frame_h - y_clipped)

                    if w_clipped <= 0 or h_clipped <= 0:
                        logger.warning(f"⚠️ Bbox outside frame for track {track_id}: ({x},{y},{w},{h}), frame_shape=({frame_h},{frame_w})")
                        continue

                    # Crop person from frame
                    person_crop = frame[y_clipped:y_clipped+h_clipped, x_clipped:x_clipped+w_clipped].copy()

                    # Validate crop is not empty
                    if person_crop.size == 0 or person_crop.shape[0] == 0 or person_crop.shape[1] == 0:
                        logger.warning(f"⚠️ Empty crop for track {track_id} at frame {frame_id}")
                        continue

                    # Check if label changed (re-verification updated it)
                    current_label = info['label']
                    saved_label = person_video_writers[track_id]['label']

                    if current_label != saved_label:
                        # Label changed - create new folder
                        logger.info(f"  🔄 Track {track_id} label changed: {saved_label} → {current_label}")

                        # Create new folder
                        person_folder = extracted_objects_dir / current_label
                        person_folder.mkdir(parents=True, exist_ok=True)

                        folder_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                        track_folder_name = f"{current_label}_{folder_timestamp}_track{track_id}"
                        track_folder_path = person_folder / track_folder_name
                        track_folder_path.mkdir(parents=True, exist_ok=True)

                        person_video_writers[track_id] = {
                            'folder': track_folder_path,
                            'label': current_label,
                            'path': track_folder_path,
                            'frame_count': 0
                        }

                        logger.info(f"  📁 Created new frames folder: {track_folder_name}")

                    # Save frame as image
                    frame_count = person_video_writers[track_id]['frame_count']
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = person_video_writers[track_id]['folder'] / frame_filename

                    # Try to write image
                    success = cv2.imwrite(str(frame_path), person_crop)
                    if not success:
                        logger.error(f"❌ Failed to write image: {frame_path}")
                    else:
                        person_video_writers[track_id]['frame_count'] += 1

                except Exception as e:
                    logger.error(f"❌ Error saving crop for track {track_id} at frame {frame_id}: {e}")
                    logger.error(f"   Bbox: ({x},{y},{w},{h}), Frame shape: {frame.shape}")
                    # Continue processing other tracks
                    continue

            # Draw on frame (ZONE-CENTRIC LOGIC)
            # Color logic:
            # Green: Person is in a zone
            # Blue: Person is outside all zones
            if zone_id:
                color = (0, 255, 0)  # Green - in a zone
            else:
                color = (255, 0, 0)  # Blue - outside zones

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Build label with time information
            label_text = f"{info['label']} (ID:{track_id})"

            # Add zone info and time (ZONE-CENTRIC LOGIC)
            if zone_id:
                label_text += f" | {zone_name} ({duration:.1f}s)"
            else:
                label_text += f" | Outside"

            # Draw label background for better visibility
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0], y), color, -1)
            cv2.putText(frame, label_text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detect lost tracks (tracks not in current frame)
        current_track_ids = set(int(track[4]) for track in tracks)
        old_track_ids = set(track_labels.keys())
        lost_tracks = old_track_ids - current_track_ids

        for lost_track_id in lost_tracks:
            old_info = track_labels[lost_track_id]
            logger.info(f"🗑️  Track {lost_track_id}: RESET (was {old_info.get('person_name', 'Unknown')}, best_sim={old_info.get('best_similarity', 0):.4f})")

            # Delete from Redis (TTL will auto-expire)
            if redis_manager:
                redis_manager.delete_track(lost_track_id)

            # Delete from in-memory
            del track_labels[lost_track_id]
            if lost_track_id in track_frame_count:
                del track_frame_count[lost_track_id]

        # Submit zone monitoring task to service thread
        # Zone service will update zone status and detect violations
        zone_task = ZoneTask(
            frame_id=frame_id,
            frame_time=frame_time,
            tracks=tracks,
            reid_results=track_labels.copy(),
            zone_ids=zone_ids_for_service.copy(),
            camera_idx=0
        )
        zone_service.submit_task(zone_task)

        # Get zone results from service (non-blocking)
        # May be from previous frame if service is still processing
        zone_result = zone_service.get_result(timeout=0.001)

        # Process violations from zone service
        if zone_result and zone_result.violations and violation_callback:
            for violation in zone_result.violations:
                violation_callback({
                    'frame_id': zone_result.frame_id,
                    'frame_time': zone_result.frame_time,
                    **violation
                })

        # Print violation table every 5 seconds
        if frame_time - zone_monitor.last_violation_table_print >= 5.0:
            zone_monitor.print_violation_table(frame_time, alert_threshold=alert_threshold)
            zone_monitor.last_violation_table_print = frame_time

        # Draw zones for all cameras (single camera = 1 iteration)
        for cam_idx in range(num_cameras):
            if num_cameras > 1:
                # Multi-camera: draw zones on each camera's portion of the combined frame
                cam_x_start = cam_idx * stream_reader.width
                cam_x_end = (cam_idx + 1) * stream_reader.width
                camera_frame = frame[:, cam_x_start:cam_x_end]
                camera_frame = zone_monitor.draw_zones(camera_frame, camera_idx=cam_idx)

                # Draw FPS for this camera (top-left corner)
                fps_text = f"Cam{cam_idx+1} FPS: {avg_fps:.1f}"
                cv2.putText(camera_frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw frame counter (below FPS)
                if is_stream:
                    frame_text = f"Frame: {frame_id}"
                else:
                    frame_text = f"Frame: {frame_id}/{total_frames}"
                cv2.putText(camera_frame, frame_text, (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                frame[:, cam_x_start:cam_x_end] = camera_frame
            else:
                # Single camera: draw zones on entire frame
                frame = zone_monitor.draw_zones(frame, camera_idx=cam_idx)

                # Draw FPS (top-left corner)
                fps_text = f"FPS: {avg_fps:.1f}"
                cv2.putText(frame, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw frame counter (below FPS)
                if is_stream:
                    frame_text = f"Frame: {frame_id}"
                else:
                    frame_text = f"Frame: {frame_id}/{total_frames}"
                cv2.putText(frame, frame_text, (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Write frame
        vid_writer.write(frame)

        # Call progress callback if provided
        if progress_callback and frame_id % 5 == 0:  # Update every 5 frames to reduce overhead
            try:
                # Prepare tracks data for callback
                tracks_data = []
                for track in tracks:
                    x1, y1, x2, y2, tid, conf = track
                    tid = int(tid)
                    info = track_labels.get(tid, {
                        'global_id': -1,
                        'similarity': 0.0,
                        'label': 'Unknown',
                        'person_name': 'Unknown'
                    })
                    tracks_data.append({
                        'track_id': tid,
                        'label': info['label'],
                        'similarity': info['similarity']
                    })
                progress_callback(frame_id, tracks_data)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

        frame_id += 1

    # Cleanup
    stream_reader.release()
    vid_writer.release()
    csv_file.close()

    # Summary of person frame folders
    logger.info("")
    logger.info("� Person frame folders summary...")
    for track_id, writer_info in person_video_writers.items():
        logger.info(f"  ✅ Track {track_id} ({writer_info['label']}): {writer_info['frame_count']} frames → {writer_info['path'].name}")

    elapsed = time.time() - start_time
    logger.info("="*80)
    # Stop zone monitoring service
    zone_service.stop()

    # Print zone service metrics
    metrics = zone_service.get_metrics()
    logger.info(f"\n📊 Zone Service Metrics:")
    logger.info(f"   Processed frames: {metrics['processed_frames']}")
    logger.info(f"   Dropped frames: {metrics['dropped_frames']}")
    logger.info(f"   Avg process time: {metrics['avg_process_time_ms']:.2f}ms")

    logger.info(f"✅ Processing completed in {elapsed:.1f}s")
    logger.info(f"   Processed {frame_id} frames @ {frame_id/elapsed:.1f} FPS")
    logger.info(f"   Person frames saved to: {extracted_objects_dir}")
    logger.info(f"   Total person frame folders: {len(person_video_writers)}")
    logger.info("="*80)

    # Print zone summary
    summary = zone_monitor.get_zone_summary()
    logger.info("\n" + "="*80)
    logger.info("📊 ZONE SUMMARY")
    logger.info("="*80)
    for zone_id, data in summary.items():
        logger.info(f"\n{data['name']} ({zone_id}):")
        logger.info(f"  Required persons: {data['required_persons']}")
        logger.info(f"  Present persons: {data['count']}")
        logger.info(f"  Status: {data['status']}")
        for person in data['present_persons']:
            logger.info(f"    ✅ {person['name']} (ID:{person['id']}) "
                       f"- {person['duration']:.1f}s")
        if data['missing_persons']:
            logger.info(f"  Missing: {data['missing_persons']}")

    # Print violations (ZONE-CENTRIC LOGIC)
    violations = zone_monitor.get_violations()
    if violations:
        logger.info("\n" + "="*80)
        logger.info("⚠️  ZONE VIOLATIONS DETECTED")
        logger.info("="*80)
        for v in violations:
            missing_str = ", ".join([f"{name} (ID:{pid})"
                                    for pid, name in zip(v['missing_persons'], v['missing_names'])])
            logger.info(f"  Zone '{v['zone_name']}' incomplete: Missing {missing_str} at {v['time']:.1f}s")

    # Save report
    zone_monitor.save_report(output_json)

    # Log Redis stats before cleanup
    if redis_manager:
        stats = redis_manager.get_stats()
        logger.info(f"\n📊 Redis Stats: {stats['total_tracks']} tracks, {stats['redis_memory_used']} memory used")

    logger.info("\n" + "="*80)
    logger.info("📁 OUTPUT FILES")
    logger.info("="*80)
    logger.info(f"  Video: {output_video}")
    logger.info(f"  CSV:   {output_csv}")
    logger.info(f"  JSON:  {output_json}")
    logger.info("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zone Monitoring with Person ReID")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--zones", type=str, default="configs/zones.yaml",
                       help="Zone configuration YAML")
    parser.add_argument("--config", type=str, default=None,
                       help="ReID config YAML (default: configs/config.yaml)")
    parser.add_argument("--similarity", type=float, default=0.8,
                       help="ReID similarity threshold (default: 0.8)")
    parser.add_argument("--iou", type=float, default=0.6,
                       help="Zone IOU threshold (default: 0.6 = 60%%)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (default: outputs/)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")

    args = parser.parse_args()

    process_video_with_zones(
        video_path=args.video,
        zone_config_path=args.zones,
        reid_config_path=args.config,
        similarity_threshold=args.similarity,
        iou_threshold=args.iou,
        output_dir=args.output,
        max_frames=args.max_frames
    )

