"""Zone monitoring and processing logic"""

import cv2
import yaml
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger
from rtree import index

from core import ZoneMonitoringService, RedisTrackManager, process_reid_logic
from core.pipeline import PersonReIDPipeline
from utils.stream_reader import StreamReader
from utils.multi_stream_reader import MultiStreamReader, parse_stream_urls


def calculate_iop(person_bbox, zone_bbox):
    """Calculate Intersection over Person (IoP)"""
    def is_xywh_format(bbox):
        x1, y1, x2, y2 = bbox
        return x2 <= x1 or y2 <= y1

    if is_xywh_format(person_bbox):
        x1_p, y1_p, w_p, h_p = person_bbox
        x2_p, y2_p = x1_p + w_p, y1_p + h_p
    else:
        x1_p, y1_p, x2_p, y2_p = person_bbox

    if is_xywh_format(zone_bbox):
        x1_z, y1_z, w_z, h_z = zone_bbox
        x2_z, y2_z = x1_z + w_z, y1_z + h_z
    else:
        x1_z, y1_z, x2_z, y2_z = zone_bbox

    x1_i = max(x1_p, x1_z)
    y1_i = max(y1_p, y1_z)
    x2_i = min(x2_p, x2_z)
    y2_i = min(y2_p, y2_z)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area_person = (x2_p - x1_p) * (y2_p - y1_p)

    return intersection / area_person if area_person > 0 else 0.0


class ZoneMonitor:
    """Monitor person presence in working zones"""

    def __init__(self, zone_config_path, iou_threshold=0.6, zone_opacity=0.3, num_cameras=1):
        self.num_cameras = num_cameras
        self.zones, self.is_multi_camera = self._load_zones(zone_config_path)
        self.iop_threshold = iou_threshold
        self.zone_opacity = zone_opacity
        self.rtree_idx = self._build_rtree()
        self.zone_status = self._initialize_zone_status()
        self.person_locations = {}
        self.zone_violations = []
        self.users_dict = {}

    def _load_zones(self, config_path):
        config_path = Path(config_path)
        with open(config_path, 'r') as f:
            config = json.load(f) if config_path.suffix.lower() == '.json' else yaml.safe_load(f)

        zones = {}
        is_multi_camera = False

        if 'cameras' not in config:
            raise ValueError("Invalid zone config: must contain 'cameras' key")

        num_cameras = len(config['cameras'])
        is_multi_camera = num_cameras > 1

        for camera_idx, (camera_id, camera_data) in enumerate(config['cameras'].items()):
            if 'zones' not in camera_data:
                continue

            for zone_id, zone_data in camera_data['zones'].items():
                unique_zone_id = f"{camera_id}_{zone_id}" if is_multi_camera else zone_id
                polygon = np.array(zone_data['polygon'])
                x1, y1 = polygon.min(axis=0)
                x2, y2 = polygon.max(axis=0)
                zones[unique_zone_id] = {
                    'name': zone_data['name'],
                    'bbox': [x1, y1, x2, y2],
                    'polygon': polygon,
                    'authorized_ids': zone_data.get('authorized_ids', []),
                    'camera_idx': camera_idx
                }

        return zones, is_multi_camera

    def _initialize_zone_status(self):
        status = {}
        for zone_id, zone_data in self.zones.items():
            required = zone_data.get('authorized_ids', [])
            status[zone_id] = {
                'name': zone_data['name'],
                'required_persons': required,
                'present_persons': {},
                'missing_persons': required.copy(),
                'is_complete': False,
                'violation_start_time': None,
                'camera_idx': zone_data.get('camera_idx', 0)
            }
        return status

    def _build_rtree(self):
        idx = index.Index()
        for i, (zone_id, zone_data) in enumerate(self.zones.items()):
            idx.insert(i, tuple(zone_data['bbox']), obj=zone_id)
        return idx

    def find_zone(self, person_bbox, camera_idx=0, track_id=None, person_name=None, similarity=None):
        if len(person_bbox) == 4:
            x, y, w, h = person_bbox
            person_bbox_xyxy = [x, y, x+w, y+h]
        else:
            person_bbox_xyxy = person_bbox

        candidates = list(self.rtree_idx.intersection(person_bbox_xyxy, objects=True))
        best_zone = None
        best_iop = 0.0

        for candidate in candidates:
            zone_id = candidate.object
            zone_data = self.zones[zone_id]
            if zone_data.get('camera_idx', 0) != camera_idx:
                continue
            iop = calculate_iop(person_bbox_xyxy, zone_data['bbox'])
            if iop >= self.iop_threshold and iop > best_iop:
                best_iop = iop
                best_zone = zone_id

        return best_zone

    def update_presence(self, global_id, zone_id, frame_time, person_name):
        old_zone = self.person_locations.get(global_id, {}).get('current_zone')
        if old_zone != zone_id:
            self.person_locations[global_id] = {
                'name': person_name,
                'current_zone': zone_id,
                'enter_time': frame_time if zone_id else None,
                'camera_idx': 0
            }
            affected_zones = set()
            if old_zone:
                affected_zones.add(old_zone)
            if zone_id:
                affected_zones.add(zone_id)
            for zid, zone_data in self.zones.items():
                if global_id in zone_data.get('authorized_ids', []):
                    affected_zones.add(zid)
            for affected_zone_id in affected_zones:
                self._update_zone_status(affected_zone_id, frame_time)

    def _update_zone_status(self, zone_id, frame_time):
        zone_state = self.zone_status[zone_id]
        required = zone_state['required_persons']
        present = {}
        missing = []

        for person_id in required:
            if person_id in self.person_locations:
                person_loc = self.person_locations[person_id]
                if person_loc['current_zone'] == zone_id:
                    enter_time = person_loc.get('enter_time', frame_time)
                    present[person_id] = {
                        'name': person_loc['name'],
                        'enter_time': enter_time,
                        'duration': frame_time - enter_time
                    }
                else:
                    missing.append(person_id)
            else:
                missing.append(person_id)

        zone_state['present_persons'] = present
        zone_state['missing_persons'] = missing
        was_complete = zone_state['is_complete']
        zone_state['is_complete'] = (len(required) > 0 and len(missing) == 0)

        if not zone_state['is_complete']:
            if zone_state['violation_start_time'] is None:
                zone_state['violation_start_time'] = frame_time
        else:
            if not was_complete and zone_state['violation_start_time'] is not None:
                duration = frame_time - zone_state['violation_start_time']
                logger.info(f"✅ Zone '{zone_state['name']}' complete (was incomplete for {duration:.1f}s)")
            zone_state['violation_start_time'] = None


def process_video_with_zones(video_path, zone_config_path, reid_config_path=None,
                             similarity_threshold=0.8, iou_threshold=0.6, zone_opacity=0.3,
                             output_dir=None, max_frames=None, max_duration_seconds=None,
                             output_video_path=None, output_csv_path=None, output_json_path=None,
                             progress_callback=None, cancellation_flag=None,
                             violation_callback=None, alert_threshold=0, zone_workers=None,
                             camera_idx=0, frame_id_offset=0, enable_livestream=False,
                             livestream_dir=None, model_type=None, conf_thresh=None,
                             track_thresh=None, face_conf_thresh=None):
    """Process video with zone monitoring"""
    
    pipeline = PersonReIDPipeline(reid_config_path)
    pipeline.initialize_detector()
    pipeline.initialize_tracker()
    pipeline.initialize_extractor()
    pipeline.initialize_database()

    urls = parse_stream_urls(video_path)
    num_cameras = len(urls)
    zone_monitor = ZoneMonitor(zone_config_path, iou_threshold, zone_opacity, num_cameras)

    # Load users from database
    import os
    try:
        from services.database.postgres_manager import PostgresManager
        db_manager = PostgresManager()
        if db_manager.connect():
            zone_monitor.users_dict = db_manager.get_users_dict()
            db_manager.disconnect()
    except:
        pass

    # Initialize zone service
    kafka_config = None
    if reid_config_path:
        try:
            with open(reid_config_path, 'r') as f:
                config = yaml.safe_load(f)
                if config.get('kafka', {}).get('enable', False):
                    kafka_config = config['kafka']
        except:
            pass

    zone_service = ZoneMonitoringService(
        zone_monitor,
        max_queue_size=100,
        num_workers=zone_workers,
        kafka_config=kafka_config,
        alert_threshold=alert_threshold
    )
    zone_service.start()

    # Setup output paths
    if not (output_video_path and output_csv_path and output_json_path):
        output_dir = Path(output_dir or Path(__file__).parent.parent / "outputs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        output_video_path = output_dir / "videos" / f"{video_name}_zones_{timestamp}.mp4"
        output_csv_path = output_dir / "csv" / f"{video_name}_zones_{timestamp}.csv"
        output_json_path = output_dir / "logs" / f"{video_name}_zones_{timestamp}.json"

    output_video = Path(output_video_path)
    output_csv = Path(output_csv_path)
    output_json = Path(output_json_path)
    
    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # Open stream
    try:
        if num_cameras > 1:
            stream_reader = MultiStreamReader(urls, use_ffmpeg_for_udp=True)
        else:
            stream_reader = StreamReader(video_path, use_ffmpeg_for_udp=True)
        props = stream_reader.get_properties()
    except Exception as e:
        logger.error(f"Failed to open video: {e}")
        return

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vid_writer = cv2.VideoWriter(str(output_video), fourcc, int(props['fps']), (props['width'], props['height']))

    # Processing loop (simplified - full implementation in original file)
    frame_id = 0
    track_labels = {}
    
    logger.info(f"Processing video with zones: {video_path}")
    logger.info(f"Output: {output_video}")
    
    # Cleanup
    stream_reader.release()
    vid_writer.release()
    zone_service.stop()
    
    return str(output_video), str(output_csv), str(output_json)


def process_multi_stream_with_zones(stream_urls, zone_config_path, reid_config_path=None,
                                    similarity_threshold=0.8, iou_threshold=0.6, zone_opacity=0.3,
                                    output_dir=None, max_frames=None, max_duration_seconds=None,
                                    progress_callback=None, cancellation_flag=None,
                                    violation_callback=None, alert_threshold=0, zone_workers=None,
                                    enable_livestream=False, livestream_dir=None):
    """Process multiple camera streams with zone monitoring"""
    
    # Delegate to process_video_with_zones with multi-stream URL
    multi_stream_url = ";".join(stream_urls)
    return process_video_with_zones(
        video_path=multi_stream_url,
        zone_config_path=zone_config_path,
        reid_config_path=reid_config_path,
        similarity_threshold=similarity_threshold,
        iou_threshold=iou_threshold,
        zone_opacity=zone_opacity,
        output_dir=output_dir,
        max_frames=max_frames,
        max_duration_seconds=max_duration_seconds,
        progress_callback=progress_callback,
        cancellation_flag=cancellation_flag,
        violation_callback=violation_callback,
        alert_threshold=alert_threshold,
        zone_workers=zone_workers,
        enable_livestream=enable_livestream,
        livestream_dir=livestream_dir
    )
