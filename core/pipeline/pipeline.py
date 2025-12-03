"""Person ReID Detection and Tracking Pipeline"""

import cv2
import csv
import yaml
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
from loguru import logger

from core.detection import TritonDetector
from core.tracking import ByteTrackWrapper
from core.reid import ArcFaceExtractor, FaceRecognitionTriton, ReIDProcessor
from core.database import QdrantVectorDB, RedisTrackManager
from core.preloaded_manager import preloaded_manager
from utils.stream_reader import StreamReader
from utils.multi_stream_reader import MultiStreamReader, parse_stream_urls


class PersonReIDPipeline:
    """Main pipeline for person detection, tracking, and re-identification"""

    def __init__(self, config_path=None):
        if preloaded_manager.is_initialized():
            logger.info("🚀 Using pre-loaded components")
            self.detector, self.tracker, self.extractor, self.database, self.config = preloaded_manager.get_components()
            self._preloaded = True
        else:
            logger.info("⏳ Using lazy loading")
            if config_path is None:
                config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.detector = None
            self.tracker = None
            self.extractor = None
            self.database = None
            self._preloaded = False

    def initialize_detector(self, model_type=None):
        if self.detector is not None:
            return
        cfg = self.config['detection']
        backend = cfg.get('backend', 'triton')
        
        if backend != 'triton':
            raise ValueError(f"Only 'triton' backend supported. Got: {backend}")
        
        from core.detection import TritonDetector
        triton_cfg = cfg['triton']
        self.detector = TritonDetector(
            triton_url=triton_cfg['url'],
            model_name=triton_cfg['model_name'],
            model_version=triton_cfg.get('model_version', ''),
            conf_thresh=cfg['conf_threshold'],
            nms_thresh=cfg['nms_threshold'],
            test_size=tuple(cfg['test_size']),
            timeout=triton_cfg.get('timeout', 10.0),
            verbose=triton_cfg.get('verbose', False)
        )

    def initialize_tracker(self):
        if self.tracker is not None:
            return
        cfg = self.config['tracking']
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg['track_thresh'],
            track_buffer=cfg['track_buffer'],
            match_thresh=cfg['match_thresh'],
            frame_rate=30,
            mot20=cfg['mot20']
        )

    def initialize_extractor(self):
        if self.extractor is not None:
            return
        cfg = self.config['reid']
        reid_backend = cfg.get('backend', 'insightface')
        if reid_backend == 'triton_pipeline':
            from core.reid import FaceRecognitionTriton
            triton_cfg = cfg.get('triton', {})
            self.extractor = FaceRecognitionTriton(
                triton_url=triton_cfg.get('url', 'localhost:8101'),
                face_detector_model=triton_cfg.get('face_detector_model', 'scrfd_10g'),
                arcface_model=triton_cfg.get('arcface_model', 'arcface_tensorrt'),
                feature_dim=triton_cfg.get('feature_dim', 512),
                face_conf_threshold=triton_cfg.get('face_conf_threshold', 0.5)
            )
        else:
            self.extractor = ArcFaceExtractor(
                model_name=cfg.get('arcface_model_name', 'buffalo_l'),
                use_cuda=cfg.get('use_cuda', True),
                feature_dim=cfg.get('feature_dim', 512),
                face_conf_thresh=cfg.get('face_conf_threshold', 0.5)
            )

    def initialize_database(self):
        if self.database is not None:
            return
        cfg = self.config['database']
        self.database = QdrantVectorDB(
            collection_name=cfg['qdrant_collection'],
            embedding_dim=cfg['embedding_dim'],
            use_grpc=cfg.get('use_grpc', False)
        )

    def process_video(self, video_path, similarity_threshold=0.8, output_dir=None,
                      output_video_path=None, output_csv_path=None, output_log_path=None,
                      max_frames=None, max_duration_seconds=None, progress_callback=None,
                      cancellation_flag=None):
        """Process video with detection, tracking, and ReID"""
        
        if not self._preloaded:
            if self.detector is None:
                self.initialize_detector()
            if self.tracker is None:
                self.initialize_tracker()
            
            # Only initialize ReID components if enabled
            enable_reid = self.config.get('reid', {}).get('enable', True)
            if enable_reid:
                logger.info("✅ ReID enabled - initializing face recognition components")
                if self.extractor is None:
                    self.initialize_extractor()
                if self.database is None:
                    self.initialize_database()
            else:
                logger.warning("⚠️ ReID disabled - all persons will be labeled as 'Unknown'")
        
        # Initialize ReID processor
        reid_processor = ReIDProcessor(self.config)

        urls = parse_stream_urls(video_path)
        
        try:
            if len(urls) > 1:
                stream_reader = MultiStreamReader(urls, use_ffmpeg_for_udp=True)
            else:
                stream_reader = StreamReader(video_path, use_ffmpeg_for_udp=True)
            props = stream_reader.get_properties()
            width, height, is_stream = props['width'], props['height'], props['is_stream']
            total_frames = 0 if is_stream else int(stream_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        except Exception as e:
            logger.error(f"Failed to open video: {e}")
            return

        if max_duration_seconds and max_frames is None:
            max_frames = int(max_duration_seconds * props['fps'])

        # Setup output paths
        if output_video_path and output_csv_path and output_log_path:
            output_video, output_csv, output_log = Path(output_video_path), Path(output_csv_path), Path(output_log_path)
        else:
            output_dir = Path(output_dir or Path(__file__).parent.parent / "outputs")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_name = f"stream_{timestamp}" if is_stream else Path(video_path).stem
            output_video = output_dir / "videos" / f"{video_name}_{timestamp}.mp4"
            output_csv = output_dir / "csv" / f"{video_name}_{timestamp}.csv"
            output_log = output_dir / "logs" / f"{video_name}_{timestamp}.log"

        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)

        extracted_objects_dir = Path(__file__).parent.parent / "outputs" / "extracted_objects"
        extracted_objects_dir.mkdir(parents=True, exist_ok=True)

        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            import os
            redis_manager = RedisTrackManager(
                job_id=job_id,
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=os.getenv('REDIS_PORT', 6379),
                ttl=os.getenv('REDIS_TTL', 300)
            )
        except:
            redis_manager = None

        person_video_writers = {}
        
        if self.config['output']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*self.config['output']['video_codec'])
            vid_writer = cv2.VideoWriter(str(output_video), fourcc, props['fps'], (width, height))
        
        csv_file = open(output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'confidence', 'global_id', 'similarity', 'label'])
        
        log_file = open(output_log, 'w')
        
        frame_id = 0
        track_labels = {}
        track_frame_count = {}
        fps_history = []
        frame_start_time = time.time()
        consecutive_failures = 0
        max_consecutive_failures = 30 if is_stream else 3

        while True:
            if cancellation_flag and cancellation_flag.is_set():
                break

            ret, frame = stream_reader.read()

            if not ret:
                if is_stream:
                    consecutive_failures += 1
                    if consecutive_failures <= max_consecutive_failures:
                        continue
                    else:
                        break
                else:
                    break

            if ret and frame is not None:
                consecutive_failures = 0

            if max_frames and frame_id >= max_frames:
                break

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections, (height, width))
            
            log_file.write(f"\n[Frame {frame_id}] Detected {len(detections)} objects, Tracked {len(tracks)} persons\n")

            for track in tracks:
                x1, y1, x2, y2, track_id, conf = track
                track_id = int(track_id)
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                if track_id not in track_frame_count:
                    track_frame_count[track_id] = 0
                track_frame_count[track_id] += 1
                current_frame_count = track_frame_count[track_id]

                # Use ReIDProcessor for centralized ReID logic
                info = reid_processor.process_track(
                    track_id=track_id,
                    frame_id=frame_id,
                    current_frame_count=current_frame_count,
                    frame=frame,
                    bbox=[x, y, w, h],
                    extractor=self.extractor,
                    database=self.database,
                    similarity_threshold=similarity_threshold,
                    redis_manager=redis_manager,
                    track_labels=track_labels,
                    log_file=log_file,
                    camera_idx=0
                )

                csv_writer.writerow([frame_id, track_id, x, y, w, h, f"{conf:.4f}", info['global_id'], f"{info['similarity']:.4f}", info['label']])

                # Save person frames
                if track_id not in person_video_writers:
                    person_label = info['label']
                    person_folder = extracted_objects_dir / person_label
                    person_folder.mkdir(parents=True, exist_ok=True)
                    folder_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    track_folder_name = f"{person_label}_{folder_timestamp}_track{track_id}"
                    track_folder_path = person_folder / track_folder_name
                    track_folder_path.mkdir(parents=True, exist_ok=True)
                    person_video_writers[track_id] = {'folder': track_folder_path, 'label': person_label, 'path': track_folder_path, 'frame_count': 0}

                if track_id in person_video_writers:
                    try:
                        if x < 0 or y < 0 or w <= 0 or h <= 0:
                            continue
                        frame_h, frame_w = frame.shape[:2]
                        x_clipped = max(0, x)
                        y_clipped = max(0, y)
                        w_clipped = min(w, frame_w - x_clipped)
                        h_clipped = min(h, frame_h - y_clipped)
                        if w_clipped <= 0 or h_clipped <= 0:
                            continue
                        person_crop = frame[y_clipped:y_clipped+h_clipped, x_clipped:x_clipped+w_clipped].copy()
                        if person_crop.size == 0:
                            continue
                        
                        current_label = info['label']
                        saved_label = person_video_writers[track_id]['label']
                        if current_label != saved_label:
                            person_folder = extracted_objects_dir / current_label
                            person_folder.mkdir(parents=True, exist_ok=True)
                            folder_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                            track_folder_name = f"{current_label}_{folder_timestamp}_track{track_id}"
                            track_folder_path = person_folder / track_folder_name
                            track_folder_path.mkdir(parents=True, exist_ok=True)
                            person_video_writers[track_id] = {'folder': track_folder_path, 'label': current_label, 'path': track_folder_path, 'frame_count': 0}

                        frame_count = person_video_writers[track_id]['frame_count']
                        frame_filename = f"frame_{frame_count:06d}.jpg"
                        frame_path = person_video_writers[track_id]['folder'] / frame_filename
                        if cv2.imwrite(str(frame_path), person_crop):
                            person_video_writers[track_id]['frame_count'] += 1
                    except Exception as e:
                        logger.error(f"Error saving crop for track {track_id}: {e}")
                        continue

                if self.config['output']['save_video']:
                    color = tuple(self.config['visualization']['color_known']) if info['label'] != 'Unknown' else tuple(self.config['visualization']['color_unknown'])
                    thickness = self.config['visualization']['bbox_thickness']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    label_text = f"{info['label']} (ID:{track_id}, sim:{info['similarity']:.2f})"
                    font_scale = self.config['visualization']['font_scale']
                    cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Lost tracks cleanup
            current_track_ids = set(int(track[4]) for track in tracks)
            lost_tracks = set(track_labels.keys()) - current_track_ids
            for lost_track_id in lost_tracks:
                if redis_manager:
                    redis_manager.delete_track(lost_track_id)
                del track_labels[lost_track_id]
                if lost_track_id in track_frame_count:
                    del track_frame_count[lost_track_id]

            # FPS calculation
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

            if self.config['output']['save_video']:
                cv2.putText(frame, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                frame_text = f"Frame: {frame_id}" if is_stream else f"Frame: {frame_id}/{total_frames}"
                cv2.putText(frame, frame_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                vid_writer.write(frame)

            if progress_callback and frame_id % 5 == 0:
                try:
                    track_info = [{'track_id': tid, 'label': info['label'], 'similarity': float(info['similarity']), 'global_id': info['global_id']} for tid, info in track_labels.items()]
                    progress_callback(frame_id, track_info)
                except:
                    pass

            frame_start_time = time.time()
            frame_id += 1

        stream_reader.release()
        if self.config['output']['save_video']:
            vid_writer.release()
        csv_file.close()
        log_file.close()

        logger.info(f"Processing complete: {frame_id} frames, {len(track_labels)} tracks, {avg_fps:.2f} FPS")
        logger.info(f"Output: {output_video}")
