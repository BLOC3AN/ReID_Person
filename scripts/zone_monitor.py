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

from core import YOLOXDetector, ByteTrackWrapper, ArcFaceExtractor, QdrantVectorDB
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
    """Monitor person presence in working zones using IoP-based overlap"""

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

        # State tracking
        self.zone_presence = {}  # {global_id: {...}}
        self.zone_history = []   # Log of all zone events
        self.last_violation_table_print = 0  # Track last time we printed violation table

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
        Update zone presence for person
        Args:
            global_id: Person's global ID
            zone_id: Current zone ID (or None)
            frame_time: Current timestamp (seconds)
            person_name: Person's name
        """
        # Initialize if new person
        if global_id not in self.zone_presence:
            # Find ALL zones this person is authorized for (support multi-camera)
            authorized_zone_ids = []
            for zid, zone_info in self.zones.items():
                if global_id in zone_info['authorized_ids']:
                    authorized_zone_ids.append(zid)

            self.zone_presence[global_id] = {
                'name': person_name,
                'current_zone': None,
                'enter_time': None,
                'total_duration': 0,
                'authorized': False,
                'authorized_zone_ids': authorized_zone_ids,  # List of zones they're authorized for
                'violations': [],
                'outside_zone_time': 0,  # Time spent outside any zone
                'outside_zone_start': frame_time,  # Start tracking outside time
                'violation_start_time': None  # Track when violation started
            }

        person = self.zone_presence[global_id]

        # Zone transition
        if person['current_zone'] != zone_id:
            # Exit old zone
            if person['current_zone']:
                duration = frame_time - person['enter_time']
                self._log_zone_event('exit', global_id, person['current_zone'],
                                    frame_time, duration)
                person['total_duration'] = 0  # Reset for new zone

            # Enter new zone
            if zone_id:
                # Update outside zone time before entering
                if 'outside_zone_start' in person and person['outside_zone_start'] is not None:
                    person['outside_zone_time'] += frame_time - person['outside_zone_start']
                    person['outside_zone_start'] = None

                person['current_zone'] = zone_id
                person['enter_time'] = frame_time

                # NEW LOGIC: authorized = True if person is in ANY of their authorized zones
                authorized_zone_ids = person.get('authorized_zone_ids', [])
                person['authorized'] = (zone_id in authorized_zone_ids)

                self._log_zone_event('enter', global_id, zone_id, frame_time, 0)

                # NEW LOGIC: Check if person is NOT in any of their authorized zones
                # Violation = person has authorized zones BUT is in different zone
                if authorized_zone_ids and zone_id not in authorized_zone_ids:
                    # Start tracking violation time
                    if person.get('violation_start_time') is None:
                        person['violation_start_time'] = frame_time

                    violation = {
                        'zone': zone_id,
                        'zone_name': self.zones[zone_id]['name'],
                        'authorized_zones': authorized_zone_ids,
                        'authorized_zone_names': [self.zones[zid]['name'] for zid in authorized_zone_ids],
                        'time': frame_time,
                        'duration': frame_time - person['violation_start_time'],
                        'type': 'not_in_authorized_zone'
                    }
                    person['violations'].append(violation)
                else:
                    # Person is in authorized zone - reset violation timer
                    person['violation_start_time'] = None
            else:
                # Left all zones - CHECK VIOLATION if person has authorized zones
                authorized_zone_ids = person.get('authorized_zone_ids', [])
                if authorized_zone_ids:
                    # Start tracking violation time
                    if person.get('violation_start_time') is None:
                        person['violation_start_time'] = frame_time

                    violation = {
                        'zone': None,
                        'zone_name': 'Outside all zones',
                        'authorized_zones': authorized_zone_ids,
                        'authorized_zone_names': [self.zones[zid]['name'] for zid in authorized_zone_ids],
                        'time': frame_time,
                        'duration': frame_time - person['violation_start_time'],
                        'type': 'outside_authorized_zone'
                    }
                    person['violations'].append(violation)
                else:
                    # Person has no authorized zones - reset violation timer
                    person['violation_start_time'] = None

                # Left all zones - start tracking outside time
                person['current_zone'] = None
                person['enter_time'] = None
                person['authorized'] = False
                person['outside_zone_start'] = frame_time

        # Update duration if in zone
        if zone_id and person['enter_time']:
            person['total_duration'] = frame_time - person['enter_time']

        # Update outside zone duration if outside
        if not zone_id and 'outside_zone_start' in person and person['outside_zone_start'] is not None:
            current_outside_duration = frame_time - person['outside_zone_start']
            # Don't accumulate yet, just track current session
    
    def _log_zone_event(self, event_type, global_id, zone_id, time, duration):
        """Log zone entry/exit events"""
        event = {
            'type': event_type,
            'global_id': global_id,
            'name': self.zone_presence[global_id]['name'],
            'zone': zone_id,
            'zone_name': self.zones[zone_id]['name'],
            'time': time,
            'duration': duration
        }
        self.zone_history.append(event)

        if event_type == 'enter':
            auth_status = "✅" if self.zone_presence[global_id]['authorized'] else "🚫"
            logger.info(f"{auth_status} {event['name']} (ID:{global_id}) entered "
                       f"'{event['zone_name']}'")
        else:
            logger.info(f"⬅️  {event['name']} (ID:{global_id}) exited "
                       f"'{event['zone_name']}' (duration: {duration:.1f}s)")

    def print_violation_table(self, current_time, alert_threshold=0):
        """
        Print violation status table for all tracked persons

        Args:
            current_time: Current frame time in seconds
            alert_threshold: Time threshold (seconds) before showing alert
        """
        table_data = []
        headers = ["User", "Current Zone", "Authorized Zones", "Status", "Duration (s)"]

        for global_id, person in self.zone_presence.items():
            name = person['name']
            current_zone = person.get('current_zone')
            authorized_zone_ids = person.get('authorized_zone_ids', [])

            # Get current zone name
            if current_zone:
                current_zone_name = self.zones[current_zone]['name']
            else:
                current_zone_name = "Outside all zones"

            # Get authorized zone names
            if authorized_zone_ids:
                auth_zone_names = ", ".join([self.zones[zid]['name'] for zid in authorized_zone_ids])
            else:
                auth_zone_names = "None"

            # Determine status
            is_authorized = person.get('authorized', False)
            violation_start = person.get('violation_start_time')

            if is_authorized:
                status = "✅ OK"
                duration = 0
            elif violation_start is not None:
                duration = current_time - violation_start
                if duration >= alert_threshold:
                    status = f"🚨 ALERT"
                else:
                    status = f"⚠️  Warning"
            else:
                status = "⚠️  Warning"
                duration = 0

            table_data.append([
                f"{name} (ID:{global_id})",
                current_zone_name,
                auth_zone_names,
                status,
                f"{duration:.1f}"
            ])

        if table_data:
            logger.info("\n" + "="*80)
            logger.info("📊 ZONE MONITORING STATUS")
            logger.info("="*80)
            logger.info("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
            logger.info("="*80 + "\n")
    
    def get_zone_summary(self):
        """Get summary of zone presence"""
        summary = {}
        for zone_id, zone_data in self.zones.items():
            persons_in_zone = [
                {
                    'id': gid, 
                    'name': p['name'], 
                    'duration': p['total_duration'],
                    'authorized': p['authorized']
                }
                for gid, p in self.zone_presence.items()
                if p['current_zone'] == zone_id
            ]
            summary[zone_id] = {
                'name': zone_data['name'],
                'authorized_ids': zone_data['authorized_ids'],
                'current_persons': persons_in_zone,
                'count': len(persons_in_zone)
            }
        return summary
    
    def get_violations(self):
        """Get all violations"""
        violations = []
        for global_id, person in self.zone_presence.items():
            for v in person['violations']:
                violations.append({
                    'global_id': global_id,
                    'name': person['name'],
                    **v
                })
        return violations
    
    def save_report(self, output_path):
        """Save zone monitoring report to JSON"""
        report = {
            'summary': self.get_zone_summary(),
            'history': self.zone_history,
            'violations': self.get_violations(),
            'zones': {
                zone_id: {
                    'name': data['name'],
                    'authorized_ids': data['authorized_ids']
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
        Draw zone boundaries on frame with border only (no background fill)

        Zone border color logic:
        - Green (0, 255, 0): All persons in zone are in their authorized zone (correct)
        - Red (0, 0, 255): At least one person in zone is NOT in their authorized zone (violation/alert)

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

            # Determine zone color based on persons in this zone
            # Default: Green (all persons authorized or no persons)
            color = (0, 255, 0)  # Green

            # Check if any person in this zone has violation
            has_violation = False
            for global_id, person_data in self.zone_presence.items():
                # Check if person is currently in this zone
                if person_data.get('current_zone') == zone_id:
                    # Check if person is NOT authorized for this zone (violation)
                    if not person_data.get('authorized', False):
                        has_violation = True
                        break

            # Set color based on violation status
            if has_violation:
                color = (0, 0, 255)  # Red - violation/alert

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

    # CSV writer
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_id', 'track_id', 'global_id', 'person_name', 'similarity',
        'x', 'y', 'w', 'h', 'zone_id', 'zone_name', 'authorized', 'duration', 'camera_idx'
    ])

    # Processing state
    frame_id = 0
    track_labels = {}
    track_frame_count = {}
    track_embeddings = {}

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

        # Get camera metadata if multi-stream
        camera_metadata = getattr(stream_reader, 'camera_metadata', None)

        # Detect
        detections = pipeline.detector.detect(frame)

        # Track
        tracks = pipeline.tracker.update(detections, (height, width))

        if frame_id % 30 == 0:
            logger.info(f"Frame {frame_id}/{total_frames}: {len(tracks)} tracks")

        # Process each track
        for track in tracks:
            x1, y1, x2, y2, track_id, conf = track
            track_id = int(track_id)

            # Convert to xywh
            x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

            # Initialize track
            if track_id not in track_frame_count:
                track_frame_count[track_id] = 0
                track_embeddings[track_id] = []

            track_frame_count[track_id] += 1
            current_frame_count = track_frame_count[track_id]

            # ReID extraction (same logic as detect_and_track.py)
            should_extract = False
            if current_frame_count <= 3:
                should_extract = True
            elif current_frame_count % 30 == 0:
                should_extract = True

            if should_extract:
                bbox = [x, y, w, h]
                embedding = pipeline.extractor.extract(frame, bbox)

                if current_frame_count <= 3:
                    track_embeddings[track_id].append(embedding)

                    if current_frame_count == 3:
                        # Voting
                        votes = {}
                        similarities = {}

                        for emb in track_embeddings[track_id]:
                            matches = pipeline.database.find_best_match(emb, threshold=0.0, top_k=1)
                            if matches:
                                gid, sim, name = matches[0]
                                key = (gid, name)
                                votes[key] = votes.get(key, 0) + 1
                                similarities[key] = max(similarities.get(key, 0), sim)

                        if votes:
                            best_key = max(votes.items(), key=lambda x: (x[1], similarities[x[0]]))[0]
                            global_id, person_name = best_key
                            similarity = similarities[best_key]

                            label = person_name if similarity >= similarity_threshold else "Unknown"

                            track_labels[track_id] = {
                                'global_id': global_id,
                                'similarity': similarity,
                                'label': label,
                                'person_name': person_name
                            }
                else:
                    # Re-verification
                    matches = pipeline.database.find_best_match(embedding, threshold=0.0, top_k=1)
                    if matches:
                        global_id, similarity, person_name = matches[0]
                        label = person_name if similarity >= similarity_threshold else "Unknown"

                        track_labels[track_id] = {
                            'global_id': global_id,
                            'similarity': similarity,
                            'label': label,
                            'person_name': person_name
                        }

            # Get track info
            info = track_labels.get(track_id, {
                'global_id': -1,
                'similarity': 0.0,
                'label': 'Unknown',
                'person_name': 'Unknown'
            })

            # ZONE MONITORING: Check which zone person is in
            person_bbox = [x, y, w, h]

            # Convert bbox to camera-relative coordinates (single camera = no conversion)
            if num_cameras > 1:
                relative_bbox, camera_idx = stream_reader.bbox_to_camera_relative(person_bbox)
            else:
                relative_bbox, camera_idx = person_bbox, 0

            # Find zone (works for both single and multi-camera)
            zone_id = zone_monitor.find_zone(relative_bbox, camera_idx)

            # Update zone presence
            if info['global_id'] > 0:
                # Store violations count before update
                old_violations_count = len(zone_monitor.zone_presence.get(info['global_id'], {}).get('violations', []))

                zone_monitor.update_presence(
                    info['global_id'],
                    zone_id,
                    frame_time,
                    info['person_name']
                )

                # Check if new violation occurred and trigger callback
                if violation_callback and info['global_id'] in zone_monitor.zone_presence:
                    person_data = zone_monitor.zone_presence[info['global_id']]
                    new_violations_count = len(person_data.get('violations', []))

                    if new_violations_count > old_violations_count:
                        # New violation occurred
                        latest_violation = person_data['violations'][-1]
                        violation_callback({
                            'global_id': info['global_id'],
                            'person_name': info['person_name'],
                            'frame_id': frame_id,
                            'frame_time': frame_time,
                            **latest_violation
                        })

            # Print violation table every 5 seconds
            if frame_time - zone_monitor.last_violation_table_print >= 5.0:
                zone_monitor.print_violation_table(frame_time, alert_threshold=alert_threshold)
                zone_monitor.last_violation_table_print = frame_time

            # Get zone info
            zone_name = zone_monitor.zones[zone_id]['name'] if zone_id else "None"
            authorized = False
            duration = 0.0

            # Get person data for time tracking
            outside_time = 0.0
            if info['global_id'] > 0 and info['global_id'] in zone_monitor.zone_presence:
                person_data = zone_monitor.zone_presence[info['global_id']]
                authorized = person_data.get('authorized', False)
                duration = person_data.get('total_duration', 0.0)
                outside_time = person_data.get('outside_zone_time', 0.0)

                # Add current outside session if currently outside
                if not zone_id and person_data.get('outside_zone_start'):
                    outside_time += frame_time - person_data['outside_zone_start']

            # Write to CSV
            csv_writer.writerow([
                frame_id, track_id, info['global_id'], info['person_name'],
                f"{info['similarity']:.4f}", x, y, w, h,
                zone_id if zone_id else "", zone_name, authorized, f"{duration:.2f}", camera_idx
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

            # Draw on frame
            # Color logic:
            # Green: Person is IN their authorized zone
            # Red: Person is NOT in their authorized zone (unauthorized or outside)
            if zone_id and authorized:
                color = (0, 255, 0)  # Green - in authorized zone
            else:
                color = (0, 0, 255)  # Red - not in authorized zone or outside

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Build label with time information
            label_text = f"{info['label']} (ID:{track_id})"

            # Add zone info and time
            if zone_id:
                label_text += f" | {zone_name} ({duration:.1f}s)"
            else:
                label_text += f" | Outside ({outside_time:.1f}s)"

            # Draw label background for better visibility
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x, y-text_size[1]-10), (x+text_size[0], y), color, -1)
            cv2.putText(frame, label_text, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Draw zones for all cameras (single camera = 1 iteration)
        for cam_idx in range(num_cameras):
            if num_cameras > 1:
                # Multi-camera: draw zones on each camera's portion of the combined frame
                cam_x_start = cam_idx * stream_reader.width
                cam_x_end = (cam_idx + 1) * stream_reader.width
                camera_frame = frame[:, cam_x_start:cam_x_end]
                camera_frame = zone_monitor.draw_zones(camera_frame, camera_idx=cam_idx)
                frame[:, cam_x_start:cam_x_end] = camera_frame
            else:
                # Single camera: draw zones on entire frame
                frame = zone_monitor.draw_zones(frame, camera_idx=cam_idx)

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
        logger.info(f"  Authorized IDs: {data['authorized_ids']}")
        logger.info(f"  Current persons: {data['count']}")
        for person in data['current_persons']:
            auth_icon = "✅" if person['authorized'] else "🚫"
            logger.info(f"    {auth_icon} {person['name']} (ID:{person['id']}) "
                       f"- {person['duration']:.1f}s")

    # Print violations
    violations = zone_monitor.get_violations()
    if violations:
        logger.info("\n" + "="*80)
        logger.info("⚠️  VIOLATIONS DETECTED")
        logger.info("="*80)
        for v in violations:
            logger.info(f"  {v['name']} (ID:{v['global_id']}) entered "
                       f"unauthorized zone '{v['zone_name']}' at {v['time']:.1f}s")

    # Save report
    zone_monitor.save_report(output_json)

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

