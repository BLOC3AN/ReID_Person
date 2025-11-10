#!/usr/bin/env python3
"""
Person ReID Detection and Tracking Pipeline
Main script for detecting, tracking, and re-identifying persons
"""

import sys
import os
import cv2
import csv
import yaml
import argparse
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import YOLOXDetector, ByteTrackWrapper, ArcFaceExtractor, QdrantVectorDB
from core.preloaded_manager import preloaded_manager
from utils.stream_reader import StreamReader
from utils.multi_stream_reader import MultiStreamReader, parse_stream_urls


class PersonReIDPipeline:
    """Main pipeline for person detection, tracking, and re-identification"""

    def __init__(self, config_path=None, use_preloaded=True):
        """
        Initialize pipeline with configuration

        Args:
            config_path: Path to config file (optional)
            use_preloaded: Use pre-loaded components if available (default: True)
        """
        self.use_preloaded = use_preloaded

        # Try to use pre-loaded components first
        if use_preloaded and preloaded_manager.is_initialized():
            logger.info("🚀 Using pre-loaded components (instant ready)")
            self.detector, self.tracker, self.extractor, self.database, self.config = preloaded_manager.get_components()
            self._preloaded = True
        else:
            # Fallback to lazy loading
            logger.info("⏳ Using lazy loading (components will be initialized on demand)")

            # Load config
            if config_path is None:
                config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)

            # Setup logging
            self.setup_logging()

            # Initialize components as None (lazy loading)
            self.detector = None
            self.tracker = None
            self.extractor = None
            self.database = None
            self._preloaded = False

        logger.info("="*80)
        logger.info("Person ReID Pipeline Initialized")
        logger.info(f"Mode: {'Pre-loaded' if self._preloaded else 'Lazy Loading'}")
        logger.info("="*80)
    
    def setup_logging(self):
        """Setup detailed logging"""
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pipeline_{timestamp}.log"
        
        # Configure loguru
        logger.remove()  # Remove default handler
        logger.add(sys.stderr, level=self.config['output']['log_level'])
        logger.add(log_file, level="DEBUG", rotation="100 MB")
        
        logger.info(f"Logging to: {log_file}")
    
    def initialize_detector(self, model_type=None):
        """Initialize YOLOX detector"""
        cfg = self.config['detection']

        # Use model_type from config if not provided
        if model_type is None:
            model_type = cfg.get('model_type', 'mot17')

        if model_type == 'mot17':
            model_path = Path(__file__).parent.parent / cfg['model_path_mot17']
        else:
            model_path = Path(__file__).parent.parent / cfg['model_path_yolox']

        self.detector = YOLOXDetector(
            model_path=str(model_path),
            model_type=model_type,
            device=cfg['device'],
            fp16=cfg['fp16'],
            conf_thresh=cfg['conf_threshold'],
            nms_thresh=cfg['nms_threshold'],
            test_size=tuple(cfg['test_size'])
        )
    
    def initialize_tracker(self):
        """Initialize ByteTrack tracker"""
        cfg = self.config['tracking']
        
        self.tracker = ByteTrackWrapper(
            track_thresh=cfg['track_thresh'],
            track_buffer=cfg['track_buffer'],
            match_thresh=cfg['match_thresh'],
            frame_rate=30,
            mot20=cfg['mot20']
        )
    
    def initialize_extractor(self):
        """Initialize ArcFace feature extractor for face recognition"""
        cfg = self.config['reid']

        logger.info("Initializing ArcFace extractor for face recognition")
        self.extractor = ArcFaceExtractor(
            model_name=cfg.get('arcface_model_name', 'buffalo_l'),
            use_cuda=cfg['use_cuda'],
            feature_dim=cfg['feature_dim']
        )
    
    def initialize_database(self):
        """Initialize Qdrant vector database"""
        cfg = self.config['database']
        
        self.database = QdrantVectorDB(
            use_qdrant=cfg['use_qdrant'],
            collection_name=cfg['qdrant_collection'],
            max_embeddings_per_person=cfg['max_embeddings_per_person'],
            embedding_dim=cfg['embedding_dim']
        )
        
        # Load existing database
        db_file = Path(__file__).parent.parent / "data" / "database" / "reid_database.pkl"
        if db_file.exists():
            logger.info(f"Loading database from {db_file}")
            self.database.load_from_file(str(db_file))
            logger.info(f"Database loaded: {self.database.get_person_count()} persons")
    
    def process_video(self, video_path, similarity_threshold=0.8, output_dir=None,
                      output_video_path=None, output_csv_path=None, output_log_path=None,
                      max_frames=None, max_duration_seconds=None, progress_callback=None,
                      cancellation_flag=None):
        """
        Process video with detection, tracking, and ReID

        Args:
            video_path: Path to input video or stream URL (e.g., 'udp://127.0.0.1:1905', 'rtsp://...')
            similarity_threshold: Cosine similarity threshold
            output_dir: Output directory for results (used if specific paths not provided)
            output_video_path: Specific path for output video (optional)
            progress_callback: Optional callback function(frame_id, tracks) for progress updates
            output_csv_path: Specific path for output CSV (optional)
            output_log_path: Specific path for output log (optional)
            max_frames: Maximum frames to process (None for all)
            max_duration_seconds: Maximum duration in seconds to process (None for all)
            cancellation_flag: Optional threading.Event() to signal cancellation
        """
        logger.info("="*80)
        logger.info(f"Processing Video: {video_path}")
        logger.info("="*80)

        # Initialize components if not pre-loaded
        if not self._preloaded:
            if self.detector is None:
                logger.info("Initializing detector...")
                self.initialize_detector()

            if self.tracker is None:
                logger.info("Initializing tracker...")
                self.initialize_tracker()

            if self.extractor is None:
                logger.info("Initializing feature extractor...")
                self.initialize_extractor()

            if self.database is None:
                logger.info("Initializing database...")
                self.initialize_database()
        else:
            logger.debug("Components already pre-loaded, skipping initialization")

        # Parse video_path to check if it contains multiple URLs
        urls = parse_stream_urls(video_path)

        # Open video/stream using appropriate reader
        try:
            if len(urls) > 1:
                # Multiple cameras - use MultiStreamReader
                logger.info(f"Detected {len(urls)} camera streams")
                stream_reader = MultiStreamReader(urls, use_ffmpeg_for_udp=True)
            else:
                # Single camera - use StreamReader
                stream_reader = StreamReader(video_path, use_ffmpeg_for_udp=True)

            props = stream_reader.get_properties()

            logger.info("="*80)
            logger.info("Stream Properties:")
            logger.info(f"  Source: {props['source']}")
            if 'sources' in props:
                for i, src in enumerate(props['sources']):
                    logger.info(f"    Camera {i+1}: {src}")
            logger.info(f"  Resolution: {props['width']}x{props['height']}")
            logger.info(f"  FPS: {props['fps']:.1f}")
            logger.info(f"  Type: {'Stream' if props['is_stream'] else 'File'}")
            logger.info(f"  Backend: {'ffmpeg subprocess' if props['using_ffmpeg'] else 'OpenCV'}")
            if 'num_streams' in props:
                logger.info(f"  Number of cameras: {props['num_streams']}")
            logger.info("="*80)

            width = props['width']
            height = props['height']
            is_stream = props['is_stream']

            # Get total frames for video files (not available for streams)
            total_frames = 0
            if not is_stream and stream_reader.cap is not None:
                total_frames = int(stream_reader.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        except Exception as e:
            logger.error(f"Failed to open video source: {e}")
            return

        # Calculate max frames from duration if specified
        if max_duration_seconds is not None and max_frames is None:
            max_frames = int(max_duration_seconds * props['fps'])
            logger.info(f"Max duration: {max_duration_seconds}s → {max_frames} frames at {props['fps']:.1f} FPS")

        # Get registered persons from database
        registered_persons = list(self.database.person_metadata.keys())
        logger.info(f"  Registered persons: {len(registered_persons)} ({registered_persons})")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        if not is_stream and total_frames > 0:
            logger.info(f"  Total frames in file: {total_frames}")
        
        # Setup output paths
        if output_video_path and output_csv_path and output_log_path:
            # Use provided paths
            output_video = Path(output_video_path)
            output_csv = Path(output_csv_path)
            output_log = Path(output_log_path)
        else:
            # Generate paths from output_dir
            if output_dir is None:
                output_dir = Path(__file__).parent.parent / "outputs"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Handle stream URL vs file path
            if is_stream:
                video_name = f"stream_{timestamp}"
            else:
                video_name = Path(video_path).stem

            output_video = output_dir / "videos" / f"{video_name}_{timestamp}.mp4"
            output_csv = output_dir / "csv" / f"{video_name}_{timestamp}.csv"
            output_log = output_dir / "logs" / f"{video_name}_{timestamp}.log"

        # Create parent directories
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        if self.config['output']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*self.config['output']['video_codec'])
            vid_writer = cv2.VideoWriter(str(output_video), fourcc, props['fps'], (width, height))
        
        # CSV writer
        csv_file = open(output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'frame_id', 'track_id', 'x', 'y', 'w', 'h', 
            'confidence', 'global_id', 'similarity', 'label'
        ])
        
        # Detailed log file
        log_file = open(output_log, 'w')
        
        # Processing loop
        frame_id = 0
        track_labels = {}  # Cache labels for each track
        track_frame_count = {}  # Count frames for each track
        track_embeddings = {}  # Store first 3 embeddings for voting

        # FPS tracking
        fps_history = []
        frame_start_time = time.time()
        avg_fps = 0.0  # Initialize avg_fps to prevent "referenced before assignment" error

        logger.info("="*80)
        logger.info("Starting Processing...")
        logger.info("="*80)

        # For streams: track consecutive read failures
        consecutive_failures = 0
        max_consecutive_failures = 30 if is_stream else 3  # Increased for streams

        while True:
            # Check cancellation flag
            if cancellation_flag is not None and cancellation_flag.is_set():
                logger.info("Processing cancelled by user")
                break

            ret, frame = stream_reader.read()

            # Handle read failures
            if not ret:
                if is_stream:
                    consecutive_failures += 1
                    if consecutive_failures <= max_consecutive_failures:
                        # For streams: SKIP failed frames immediately, don't wait
                        # This prevents blocking when stream is lagging
                        logger.debug(f"Skipped frame (failure {consecutive_failures}/{max_consecutive_failures})")
                        continue
                    else:
                        logger.error(f"Stream ended or lost connection after {consecutive_failures} consecutive failures")
                        break
                else:
                    # For video files, end of file is normal
                    logger.info("Reached end of video file")
                    break

            # Reset failure counter on successful read
            if ret and frame is not None:
                consecutive_failures = 0

            # Check max frames limit
            if max_frames is not None and frame_id >= max_frames:
                logger.info(f"Reached max frames limit: {max_frames}")
                break

            # Detect
            detections = self.detector.detect(frame)
            
            # Track
            tracks = self.tracker.update(detections, (height, width))
            
            # Log frame info
            log_msg = f"\n[Frame {frame_id}] Detected {len(detections)} objects, Tracked {len(tracks)} persons\n"
            log_file.write(log_msg)

            if frame_id % 30 == 0:
                if is_stream:
                    logger.info(f"Frame {frame_id}: {len(tracks)} tracks")
                else:
                    logger.info(f"Frame {frame_id}/{total_frames}: {len(tracks)} tracks")
            
            # Process each track
            for track in tracks:
                x1, y1, x2, y2, track_id, conf = track
                track_id = int(track_id)

                # Convert to xywh
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)

                # Initialize track frame count
                if track_id not in track_frame_count:
                    track_frame_count[track_id] = 0
                    track_embeddings[track_id] = []

                track_frame_count[track_id] += 1
                current_frame_count = track_frame_count[track_id]

                # Strategy: First-3 frames + Re-verify every 30 frames
                should_extract = False

                if current_frame_count <= 3:
                    # First 3 frames: collect embeddings for voting
                    should_extract = True
                    reason = f"first-{current_frame_count}"
                elif current_frame_count % 30 == 0:
                    # Re-verify every 30 frames (1 second at 30fps)
                    should_extract = True
                    reason = "re-verify"

                if should_extract:
                    bbox = [x, y, w, h]
                    embedding = self.extractor.extract(frame, bbox)

                    if current_frame_count <= 3:
                        # Store embedding for voting
                        track_embeddings[track_id].append(embedding)

                        # After 3rd frame, perform majority voting
                        if current_frame_count == 3:
                            # Match all 3 embeddings
                            votes = {}  # {(global_id, name): count}
                            similarities = {}  # {(global_id, name): max_similarity}

                            for emb in track_embeddings[track_id]:
                                matches = self.database.find_best_match(emb, threshold=0.0, top_k=1)
                                if matches:
                                    gid, sim, name = matches[0]
                                    key = (gid, name)
                                    votes[key] = votes.get(key, 0) + 1
                                    similarities[key] = max(similarities.get(key, 0), sim)

                            # Get majority vote
                            if votes:
                                best_key = max(votes.items(), key=lambda x: (x[1], similarities[x[0]]))[0]
                                global_id, person_name = best_key
                                similarity = similarities[best_key]

                                # Determine label
                                if similarity >= similarity_threshold:
                                    label = person_name
                                else:
                                    label = "Unknown"

                                track_labels[track_id] = {
                                    'global_id': global_id,
                                    'similarity': similarity,
                                    'label': label,
                                    'person_name': person_name,
                                    'votes': votes[best_key]
                                }

                                log_msg = f"  Track {track_id} [VOTING]: {votes[best_key]}/3 votes → " \
                                         f"{label} (sim={similarity:.4f}, gid={global_id})\n"
                                log_file.write(log_msg)
                                logger.info(f"  Track {track_id}: {label} (votes={votes[best_key]}/3, sim={similarity:.4f})")

                    else:
                        # Re-verification
                        matches = self.database.find_best_match(embedding, threshold=0.0, top_k=1)
                        if matches:
                            global_id, similarity, person_name = matches[0]

                            # Update only if confidence is high or current label is Unknown
                            old_label = track_labels.get(track_id, {}).get('label', 'Unknown')

                            if similarity >= similarity_threshold:
                                new_label = person_name
                            else:
                                new_label = "Unknown"

                            # Update if changed
                            if new_label != old_label or similarity >= similarity_threshold:
                                track_labels[track_id] = {
                                    'global_id': global_id,
                                    'similarity': similarity,
                                    'label': new_label,
                                    'person_name': person_name
                                }

                                log_msg = f"  Track {track_id} [RE-VERIFY]: {old_label} → {new_label} " \
                                         f"(sim={similarity:.4f}, frame={current_frame_count})\n"
                                log_file.write(log_msg)
                                logger.info(f"  Track {track_id}: Re-verified {old_label} → {new_label} (sim={similarity:.4f})")
                
                # Get cached label
                info = track_labels.get(track_id, {
                    'global_id': -1,
                    'similarity': 0.0,
                    'label': 'Unknown'
                })
                
                # Write to CSV
                csv_writer.writerow([
                    frame_id, track_id, x, y, w, h, f"{conf:.4f}",
                    info['global_id'], f"{info['similarity']:.4f}", info['label']
                ])
                
                # Draw on frame
                if self.config['output']['save_video']:
                    # Use green for known persons, red for unknown
                    color = tuple(self.config['visualization']['color_known']) \
                           if info['label'] != 'Unknown' \
                           else tuple(self.config['visualization']['color_unknown'])

                    thickness = self.config['visualization']['bbox_thickness']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

                    # Label text
                    label_text = f"{info['label']} (ID:{track_id}, sim:{info['similarity']:.2f})"
                    font_scale = self.config['visualization']['font_scale']
                    cv2.putText(frame, label_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)

            # Calculate FPS
            frame_end_time = time.time()
            frame_time = frame_end_time - frame_start_time
            current_fps = 1.0 / frame_time if frame_time > 0 else 0
            fps_history.append(current_fps)

            # Keep only last 30 frames for moving average
            if len(fps_history) > 30:
                fps_history.pop(0)

            avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0

            # Draw FPS on frame
            if self.config['output']['save_video']:
                # FPS text (top-left corner)
                fps_text = f"FPS: {avg_fps:.2f}"
                cv2.putText(frame, fps_text, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

                # Frame counter (top-left, below FPS)
                if is_stream:
                    frame_text = f"Frame: {frame_id}"
                else:
                    frame_text = f"Frame: {frame_id}/{total_frames}"
                cv2.putText(frame, frame_text, (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                vid_writer.write(frame)

            # Call progress callback if provided
            if progress_callback and frame_id % 5 == 0:  # Update every 5 frames to reduce overhead
                try:
                    # Prepare track info for callback
                    track_info = []
                    for track_id, info in track_labels.items():
                        track_info.append({
                            'track_id': track_id,
                            'label': info['label'],
                            'similarity': float(info['similarity']),
                            'global_id': info['global_id']
                        })
                    progress_callback(frame_id, track_info)
                except Exception as e:
                    logger.warning(f"Progress callback error: {e}")

            # Reset timer for next frame
            frame_start_time = time.time()
            frame_id += 1

        # Cleanup
        stream_reader.release()
        if self.config['output']['save_video']:
            vid_writer.release()
        csv_file.close()
        log_file.close()
        
        logger.info("="*80)
        logger.info("Processing Complete!")
        logger.info("="*80)
        logger.info(f"Output video: {output_video}")
        logger.info(f"Output CSV: {output_csv}")
        logger.info(f"Detailed log: {output_log}")
        logger.info(f"Total frames processed: {frame_id}")
        logger.info(f"Total tracks: {len(track_labels)}")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info("")
        logger.info("Track Summary:")
        for tid, info in sorted(track_labels.items()):
            votes_info = f" (votes={info['votes']}/3)" if 'votes' in info else ""
            logger.info(f"  Track {tid}: {info['label']} (sim={info['similarity']:.4f}, gid={info['global_id']}){votes_info}")
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Person ReID Detection and Tracking")
    parser.add_argument("--video", type=str, required=True,
                       help="Input video path or stream URL (e.g., udp://127.0.0.1:1905)")
    parser.add_argument("--model", type=str, default="mot17",
                       choices=['mot17', 'yolox'], help="Detection model")
    parser.add_argument("--threshold", type=float, default=0.8,
                       help="Similarity threshold (0.0-1.0)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (for testing)")
    parser.add_argument("--max-duration", type=int, default=None,
                       help="Maximum duration in seconds to process (for streams)")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = PersonReIDPipeline(config_path=args.config)

    # Initialize components
    pipeline.initialize_detector(model_type=args.model)
    pipeline.initialize_tracker()
    pipeline.initialize_extractor()
    pipeline.initialize_database()

    # Process video
    pipeline.process_video(
        video_path=args.video,
        similarity_threshold=args.threshold,
        output_dir=args.output,
        max_frames=args.max_frames,
        max_duration_seconds=args.max_duration
    )


if __name__ == "__main__":
    main()

