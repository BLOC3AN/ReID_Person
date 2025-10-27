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
import numpy as np
from pathlib import Path
from datetime import datetime
from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import YOLOXDetector, ByteTrackWrapper, OSNetExtractor, QdrantVectorDB


class PersonReIDPipeline:
    """Main pipeline for person detection, tracking, and re-identification"""
    
    def __init__(self, config_path=None):
        """Initialize pipeline with configuration"""
        # Load config
        if config_path is None:
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize components
        self.detector = None
        self.tracker = None
        self.extractor = None
        self.database = None
        
        logger.info("="*80)
        logger.info("Person ReID Pipeline Initialized")
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
    
    def initialize_detector(self, model_type='mot17'):
        """Initialize YOLOX detector"""
        cfg = self.config['detection']
        
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
        """Initialize OSNet feature extractor"""
        cfg = self.config['reid']
        
        self.extractor = OSNetExtractor(
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
    
    def process_video(self, video_path, known_person_name="Khiem",
                     similarity_threshold=0.8, output_dir=None, max_frames=None):
        """
        Process video with detection, tracking, and ReID

        Args:
            video_path: Path to input video
            known_person_name: Name of known person to match
            similarity_threshold: Cosine similarity threshold
            output_dir: Output directory for results
            max_frames: Maximum frames to process (None for all)
        """
        logger.info("="*80)
        logger.info(f"Processing Video: {video_path}")
        logger.info("="*80)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video Info:")
        logger.info(f"  Resolution: {width}x{height}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Total frames: {total_frames}")
        logger.info(f"  Known person: {known_person_name}")
        logger.info(f"  Similarity threshold: {similarity_threshold}")
        
        # Setup output
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "outputs"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Output paths
        video_name = Path(video_path).stem
        output_video = output_dir / "videos" / f"{video_name}_{timestamp}.mp4"
        output_csv = output_dir / "csv" / f"{video_name}_{timestamp}.csv"
        output_log = output_dir / "logs" / f"{video_name}_{timestamp}.log"
        
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        output_log.parent.mkdir(parents=True, exist_ok=True)
        
        # Video writer
        if self.config['output']['save_video']:
            fourcc = cv2.VideoWriter_fourcc(*self.config['output']['video_codec'])
            vid_writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        
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
        
        logger.info("="*80)
        logger.info("Starting Processing...")
        logger.info("="*80)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                logger.info(f"Frame {frame_id}/{total_frames}: {len(tracks)} tracks")
            
            # Process each track
            for track in tracks:
                x1, y1, x2, y2, track_id, conf = track
                track_id = int(track_id)
                
                # Convert to xywh
                x, y, w, h = int(x1), int(y1), int(x2-x1), int(y2-y1)
                
                # Extract embedding and match (only once per track)
                if track_id not in track_labels:
                    bbox = [x, y, w, h]
                    embedding = self.extractor.extract(frame, bbox)
                    
                    # Find best match
                    matches = self.database.find_best_match(
                        embedding, 
                        threshold=0.0,  # Get best match regardless
                        top_k=1
                    )
                    
                    if matches:
                        global_id, similarity = matches[0]
                        
                        # Determine label
                        if similarity >= similarity_threshold and global_id == 1:
                            label = known_person_name
                        else:
                            label = "Unknown"
                        
                        track_labels[track_id] = {
                            'global_id': global_id,
                            'similarity': similarity,
                            'label': label
                        }
                        
                        # Detailed log
                        log_msg = f"  Track {track_id}: bbox=[{x},{y},{w},{h}], " \
                                 f"similarity={similarity:.4f}, global_id={global_id} → {label}\n"
                        log_file.write(log_msg)
                        
                        if frame_id < 10 or frame_id % 100 == 0:
                            logger.info(f"  Track {track_id}: {label} (sim={similarity:.4f})")
                
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
                    color = tuple(self.config['visualization']['color_known']) \
                           if info['label'] == known_person_name \
                           else tuple(self.config['visualization']['color_unknown'])
                    
                    thickness = self.config['visualization']['bbox_thickness']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
                    
                    # Label text
                    label_text = f"{info['label']} (ID:{track_id}, sim:{info['similarity']:.2f})"
                    font_scale = self.config['visualization']['font_scale']
                    cv2.putText(frame, label_text, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
            
            # Write frame
            if self.config['output']['save_video']:
                vid_writer.write(frame)
            
            frame_id += 1
        
        # Cleanup
        cap.release()
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
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="Person ReID Detection and Tracking")
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--model", type=str, default="mot17", 
                       choices=['mot17', 'yolox'], help="Detection model")
    parser.add_argument("--known-person", type=str, default="Khiem", 
                       help="Known person name")
    parser.add_argument("--threshold", type=float, default=0.8, 
                       help="Similarity threshold")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output directory")
    parser.add_argument("--config", type=str, default=None,
                       help="Config file path")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process (for testing)")

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
        known_person_name=args.known_person,
        similarity_threshold=args.threshold,
        output_dir=args.output,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()

