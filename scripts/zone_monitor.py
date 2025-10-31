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


def calculate_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IOU) between two bboxes
    Args:
        bbox1: [x1, y1, x2, y2] or [x, y, w, h] format
        bbox2: [x1, y1, x2, y2] or [x, y, w, h] format
    Returns:
        IOU value (0-1)
    """
    # Auto-detect format: if x2 <= x1 or y2 <= y1, it's likely xywh
    # Otherwise, check if w,h values are reasonable (> coordinates)
    def is_xywh_format(bbox):
        x1, y1, x2, y2 = bbox
        # If x2 <= x1 or y2 <= y1, definitely xywh
        if x2 <= x1 or y2 <= y1:
            return True
        # If x2-x1 and y2-y1 are very large compared to x1,y1, likely xyxy
        # Otherwise might be xywh with large coordinates
        return False

    # Convert bbox1
    if is_xywh_format(bbox1):
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
    else:
        x1_1, y1_1, x2_1, y2_1 = bbox1

    # Convert bbox2
    if is_xywh_format(bbox2):
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
    else:
        x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


class ZoneMonitor:
    """Monitor person presence in working zones using IOU-based overlap"""
    
    def __init__(self, zone_config_path, iou_threshold=0.6):
        """
        Args:
            zone_config_path: Path to zone configuration YAML
            iou_threshold: IOU threshold for zone overlap (default: 0.6 = 60%)
        """
        self.zones = self._load_zones(zone_config_path)
        self.iou_threshold = iou_threshold
        self.rtree_idx = self._build_rtree()
        
        # State tracking
        self.zone_presence = {}  # {global_id: {...}}
        self.zone_history = []   # Log of all zone events
        
        logger.info(f"✅ ZoneMonitor initialized with {len(self.zones)} zones")
        logger.info(f"   IOU threshold: {iou_threshold*100:.0f}%")
    
    def _load_zones(self, config_path):
        """Load zone definitions from YAML"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        zones = {}
        for zone_id, zone_data in config['zones'].items():
            # Convert polygon to bbox [x1, y1, x2, y2]
            polygon = np.array(zone_data['polygon'])
            x1, y1 = polygon.min(axis=0)
            x2, y2 = polygon.max(axis=0)
            
            zones[zone_id] = {
                'name': zone_data['name'],
                'bbox': [x1, y1, x2, y2],
                'polygon': polygon,
                'authorized_ids': zone_data.get('authorized_ids', [])
            }
        
        return zones
    
    def _build_rtree(self):
        """Build R-tree spatial index for zones"""
        idx = index.Index()
        for i, (zone_id, zone_data) in enumerate(self.zones.items()):
            bbox = zone_data['bbox']
            # R-tree format: (minx, miny, maxx, maxy)
            idx.insert(i, tuple(bbox), obj=zone_id)
        
        logger.info(f"   R-tree index built for {len(self.zones)} zones")
        return idx
    
    def find_zone(self, person_bbox):
        """
        Find which zone contains the person using IOU >= threshold
        Args:
            person_bbox: [x1, y1, x2, y2] or [x, y, w, h]
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
        
        # Find best IOU match
        best_zone = None
        best_iou = 0.0
        
        for candidate in candidates:
            zone_id = candidate.object
            zone_bbox = self.zones[zone_id]['bbox']
            
            iou = calculate_iou(person_bbox_xyxy, zone_bbox)
            
            if iou >= self.iou_threshold and iou > best_iou:
                best_iou = iou
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
            self.zone_presence[global_id] = {
                'name': person_name,
                'current_zone': None,
                'enter_time': None,
                'total_duration': 0,
                'authorized': False,
                'violations': []
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
                person['current_zone'] = zone_id
                person['enter_time'] = frame_time
                person['authorized'] = global_id in self.zones[zone_id]['authorized_ids']
                
                self._log_zone_event('enter', global_id, zone_id, frame_time, 0)
                
                # Check authorization
                if not person['authorized']:
                    violation = {
                        'zone': zone_id,
                        'zone_name': self.zones[zone_id]['name'],
                        'time': frame_time,
                        'type': 'unauthorized_entry'
                    }
                    person['violations'].append(violation)
                    logger.warning(f"⚠️  VIOLATION: {person_name} (ID:{global_id}) "
                                 f"entered unauthorized zone '{self.zones[zone_id]['name']}'")
            else:
                # Left all zones
                person['current_zone'] = None
                person['enter_time'] = None
                person['authorized'] = False
        
        # Update duration if in zone
        if zone_id and person['enter_time']:
            person['total_duration'] = frame_time - person['enter_time']
    
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

    def draw_zones(self, frame):
        """Draw zone boundaries on frame"""
        # Draw ruler first (background layer)
        frame = self.draw_ruler(frame)

        # Draw zones on top
        for zone_id, zone_data in self.zones.items():
            polygon = zone_data['polygon']
            pts = polygon.reshape((-1, 1, 2)).astype(np.int32)

            # Draw polygon
            cv2.polylines(frame, [pts], isClosed=True, color=(255, 255, 0), thickness=2)

            # Draw zone name
            x, y = polygon[0]
            cv2.putText(frame, zone_data['name'], (int(x), int(y)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return frame


def process_video_with_zones(video_path, zone_config_path, reid_config_path=None,
                             similarity_threshold=0.8, iou_threshold=0.6,
                             output_dir=None, max_frames=None,
                             output_video_path=None, output_csv_path=None, output_json_path=None,
                             progress_callback=None):
    """
    Process video with zone monitoring integrated into ReID pipeline

    Args:
        video_path: Path to input video
        zone_config_path: Path to zone configuration YAML
        reid_config_path: Path to ReID config (default: configs/config.yaml)
        similarity_threshold: ReID similarity threshold
        iou_threshold: Zone IOU threshold (default: 0.6 = 60%)
        output_dir: Output directory for results (used if specific paths not provided)
        max_frames: Maximum frames to process
        output_video_path: Specific path for output video (optional)
        output_csv_path: Specific path for output CSV (optional)
        output_json_path: Specific path for output JSON (optional)
        progress_callback: Optional callback function(frame_id, tracks) for progress updates
    """
    from detect_and_track import PersonReIDPipeline

    logger.info("="*80)
    logger.info("🎯 ZONE MONITORING WITH PERSON REID")
    logger.info("="*80)

    # Initialize ReID pipeline
    pipeline = PersonReIDPipeline(reid_config_path)
    pipeline.initialize_detector()
    pipeline.initialize_tracker()
    pipeline.initialize_extractor()
    pipeline.initialize_database()

    # Initialize zone monitor
    zone_monitor = ZoneMonitor(zone_config_path, iou_threshold)

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

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video: {video_path}")
    logger.info(f"Resolution: {width}x{height} @ {fps} FPS")
    logger.info(f"Total frames: {total_frames}")

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))

    # CSV writer
    csv_file = open(output_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'frame_id', 'track_id', 'global_id', 'person_name', 'similarity',
        'x', 'y', 'w', 'h', 'zone_id', 'zone_name', 'authorized', 'duration'
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
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames and frame_id >= max_frames:
            logger.info(f"Reached max frames limit: {max_frames}")
            break

        frame_time = frame_id / fps  # Time in seconds

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
            zone_id = zone_monitor.find_zone(person_bbox)

            # Update zone presence
            if info['global_id'] > 0:
                zone_monitor.update_presence(
                    info['global_id'],
                    zone_id,
                    frame_time,
                    info['person_name']
                )

            # Get zone info
            zone_name = zone_monitor.zones[zone_id]['name'] if zone_id else "None"
            authorized = False
            duration = 0.0

            if info['global_id'] > 0 and info['global_id'] in zone_monitor.zone_presence:
                person_data = zone_monitor.zone_presence[info['global_id']]
                authorized = person_data.get('authorized', False)
                duration = person_data.get('total_duration', 0.0)

            # Write to CSV
            csv_writer.writerow([
                frame_id, track_id, info['global_id'], info['person_name'],
                f"{info['similarity']:.4f}", x, y, w, h,
                zone_id if zone_id else "", zone_name, authorized, f"{duration:.2f}"
            ])

            # Draw on frame
            # Color: Green=authorized, Red=unauthorized, Gray=unknown
            if info['label'] == 'Unknown':
                color = (128, 128, 128)  # Gray
            elif authorized:
                color = (0, 255, 0)  # Green
            else:
                color = (0, 0, 255)  # Red

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Label
            label_text = f"{info['label']} (ID:{track_id})"
            if zone_id:
                label_text += f" | {zone_name}"

            cv2.putText(frame, label_text, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw zones
        frame = zone_monitor.draw_zones(frame)

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
    cap.release()
    vid_writer.release()
    csv_file.close()

    elapsed = time.time() - start_time
    logger.info("="*80)
    logger.info(f"✅ Processing completed in {elapsed:.1f}s")
    logger.info(f"   Processed {frame_id} frames @ {frame_id/elapsed:.1f} FPS")
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

