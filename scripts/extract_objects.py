#!/usr/bin/env python3
"""
Extract individual object videos from tracking results
Uses MOT17 detector + ByteTrack tracker to extract person objects from video
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import argparse
import numpy as np
from collections import defaultdict
from loguru import logger
from core import YOLOXDetector, ByteTrackWrapper


def extract_objects_from_video(
    video_path: str,
    output_dir: str = "./output_objects",
    model_type: str = "mot17",
    padding: int = 10,
    conf_thresh: float = 0.6,
    track_thresh: float = 0.5,
    min_frames: int = 10
):
    """
    Extract individual object videos from source video using MOT17 + ByteTrack
    
    Args:
        video_path: Path to source video
        output_dir: Output directory for extracted object videos
        model_type: Model type ('mot17' or 'yolox')
        padding: Padding pixels around bounding box
        conf_thresh: Detection confidence threshold
        track_thresh: Tracking confidence threshold
        min_frames: Minimum frames required to save an object
    """
    
    logger.info("=" * 80)
    logger.info("EXTRACTING OBJECTS FROM VIDEO")
    logger.info("=" * 80)
    logger.info(f"Video: {video_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Model: {model_type}")
    logger.info(f"Padding: {padding}px")
    logger.info(f"Min frames: {min_frames}")
    logger.info("=" * 80)
    
    # Initialize detector
    logger.info("\nInitializing detector...")
    if model_type == "mot17":
        model_path = Path(__file__).parent.parent / "models" / "bytetrack_x_mot17.pth.tar"
    else:
        model_path = Path(__file__).parent.parent / "models" / "yolox_x.pth"
    
    detector = YOLOXDetector(
        model_path=str(model_path),
        model_type=model_type,
        conf_thresh=conf_thresh,
        nms_thresh=0.45
    )
    
    # Initialize tracker
    logger.info("Initializing tracker...")
    tracker = ByteTrackWrapper(
        track_thresh=track_thresh,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30
    )
    
    # Open video
    logger.info(f"\nOpening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Resolution: {video_width}x{video_height}")
    logger.info(f"FPS: {fps}")
    logger.info(f"Total frames: {total_frames}")
    
    # Read all frames
    logger.info("\nReading video frames...")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    logger.info(f"✅ Loaded {len(frames)} frames")
    
    # Track objects
    logger.info("\nTracking objects...")
    tracks = defaultdict(list)  # {track_id: [(frame_id, x, y, w, h), ...]}
    
    for frame_id, frame in enumerate(frames):
        # Detect
        detections = detector.detect(frame)
        
        # Track
        online_targets = tracker.update(detections, frame.shape[:2])
        
        # Store tracking results
        for track in online_targets:
            track_id = track.track_id
            bbox = track.tlbr  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            x, y, w, h = x1, y1, x2 - x1, y2 - y1
            tracks[track_id].append((frame_id, int(x), int(y), int(w), int(h)))
        
        if (frame_id + 1) % 100 == 0:
            logger.info(f"  Processed {frame_id + 1}/{len(frames)} frames")
    
    logger.info(f"✅ Found {len(tracks)} unique objects")
    
    # Filter tracks by minimum frames
    valid_tracks = {tid: dets for tid, dets in tracks.items() if len(dets) >= min_frames}
    logger.info(f"✅ {len(valid_tracks)} objects with >= {min_frames} frames")
    
    if len(valid_tracks) == 0:
        logger.warning("No valid tracks found!")
        return
    
    # Create output directory
    video_name = Path(video_path).stem
    output_base = Path(output_dir) / video_name
    output_base.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nOutput directory: {output_base}")
    
    # Extract object videos
    logger.info("\nExtracting object videos...")
    for track_id, detections in valid_tracks.items():
        logger.info(f"Processing object ID: {track_id} ({len(detections)} frames)")
        
        # Sort by frame_id
        detections.sort(key=lambda x: x[0])
        
        # Determine max bbox size for consistent video dimensions
        max_w = max(d[3] for d in detections) + 2 * padding
        max_h = max(d[4] for d in detections) + 2 * padding
        
        # Create video writer
        output_path = output_base / f"object_{track_id}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (max_w, max_h))
        
        # Create detection map
        detection_map = {d[0]: d for d in detections}
        
        # Get frame range
        start_frame = detections[0][0]
        end_frame = detections[-1][0]
        
        # Extract frames
        for frame_id in range(start_frame, end_frame + 1):
            if frame_id >= len(frames):
                break
            
            frame = frames[frame_id]
            
            if frame_id in detection_map:
                # Get bbox
                _, x, y, w, h = detection_map[frame_id]
                
                # Calculate crop coordinates with padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(video_width, x + w + padding)
                y2 = min(video_height, y + h + padding)
                
                # Crop object
                cropped = frame[y1:y2, x1:x2]
                
                # Resize to fixed size
                if cropped.shape[0] > 0 and cropped.shape[1] > 0:
                    canvas = cv2.resize(cropped, (max_w, max_h))
                    out.write(canvas)
            else:
                # Write black frame if object not detected
                black_frame = np.zeros((max_h, max_w, 3), dtype=np.uint8)
                out.write(black_frame)
        
        out.release()
        logger.info(f"  ✅ Saved: {output_path} ({end_frame - start_frame + 1} frames)")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ EXTRACTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total objects extracted: {len(valid_tracks)}")
    logger.info(f"Output directory: {output_base}")
    logger.info("=" * 80)
    logger.info("\nNext step:")
    logger.info("  Use extracted videos to register persons:")
    logger.info(f"  python scripts/register_mot17.py --video {output_base}/object_X.mp4 --name <NAME>")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract individual object videos from tracking results"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to source video file"
    )
    parser.add_argument(
        "--output-dir",
        default="./output_objects",
        help="Output directory for extracted object videos (default: ./output_objects)"
    )
    parser.add_argument(
        "--model",
        choices=["mot17", "yolox"],
        default="mot17",
        help="Model type to use (default: mot17)"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=10,
        help="Padding pixels around bounding box (default: 10)"
    )
    parser.add_argument(
        "--conf-thresh",
        type=float,
        default=0.6,
        help="Detection confidence threshold (default: 0.6)"
    )
    parser.add_argument(
        "--track-thresh",
        type=float,
        default=0.5,
        help="Tracking confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=10,
        help="Minimum frames required to save an object (default: 10)"
    )
    
    args = parser.parse_args()
    
    extract_objects_from_video(
        video_path=args.video,
        output_dir=args.output_dir,
        model_type=args.model,
        padding=args.padding,
        conf_thresh=args.conf_thresh,
        track_thresh=args.track_thresh,
        min_frames=args.min_frames
    )

