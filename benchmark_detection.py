#!/usr/bin/env python3
"""
Benchmark Detection Pipeline - Detailed Performance Analysis
Measures: FPS, GPU usage, CPU usage, memory, frame drops, bottlenecks
"""

import sys
import os
import cv2
import time
import yaml
import psutil
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
from loguru import logger

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False
    logger.warning("pynvml not available - GPU monitoring disabled")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core.detector_triton import TritonDetector
from core.tracker import ByteTrackWrapper
from core.feature_extractor import ArcFaceExtractor
from core.vector_db import QdrantVectorDB


class PerformanceMonitor:
    """Monitor system performance in real-time"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.running = False
        self.metrics = {
            'gpu_util': deque(maxlen=100),
            'gpu_mem': deque(maxlen=100),
            'cpu_util': deque(maxlen=100),
            'ram_util': deque(maxlen=100),
        }
        
        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    def start(self):
        """Start monitoring in background thread"""
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            # CPU & RAM
            self.metrics['cpu_util'].append(psutil.cpu_percent(interval=0.1))
            self.metrics['ram_util'].append(psutil.virtual_memory().percent)
            
            # GPU
            if GPU_AVAILABLE:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    self.metrics['gpu_util'].append(util.gpu)
                    self.metrics['gpu_mem'].append(mem.used / mem.total * 100)
                except:
                    pass
            
            time.sleep(0.5)
    
    def get_stats(self):
        """Get current statistics"""
        stats = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                stats[f'{key}_avg'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
        return stats


class DetectionBenchmark:
    """Benchmark detection pipeline with detailed metrics"""
    
    def __init__(self, video_path, config_path="configs/config.yaml"):
        self.video_path = video_path

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        logger.info("🔧 Initializing components...")
        self._init_components()
        
        # Performance tracking
        self.frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.tracking_times = deque(maxlen=100)
        self.reid_times = deque(maxlen=100)
        self.total_frames = 0
        self.dropped_frames = 0
        
        # System monitor
        self.monitor = PerformanceMonitor(gpu_id=0)
    
    def _init_components(self):
        """Initialize detection components"""
        cfg = self.config['detection']
        
        # Detector (Triton)
        if cfg['backend'] == 'triton':
            logger.info("📡 Using Triton Inference Server")
            triton_cfg = cfg['triton']
            self.detector = TritonDetector(
                triton_url=triton_cfg['url'],
                model_name=triton_cfg['model_name'],
                model_version=triton_cfg.get('model_version', ''),
                conf_thresh=cfg['conf_threshold'],
                nms_thresh=cfg['nms_threshold'],
                test_size=tuple(cfg['test_size']),
                timeout=triton_cfg.get('timeout', 10.0)
            )
        else:
            raise ValueError(f"Only Triton backend supported in benchmark")
        
        # Tracker
        track_cfg = self.config['tracking']
        self.tracker = ByteTrackWrapper(
            track_thresh=track_cfg['track_thresh'],
            track_buffer=track_cfg['track_buffer'],
            match_thresh=track_cfg['match_thresh']
        )
        
        # Feature extractor (ArcFace)
        reid_cfg = self.config['reid']
        self.extractor = ArcFaceExtractor(
            model_name=reid_cfg['arcface_model_name'],
            use_cuda=reid_cfg['use_cuda']
        )
        
        # Database
        db_cfg = self.config['database']
        self.database = QdrantVectorDB(
            collection_name=db_cfg['qdrant_collection'],
            embedding_dim=db_cfg['embedding_dim']
        )
        
        logger.info("✅ All components initialized")
    
    def run(self, max_frames=None, show_video=False):
        """Run benchmark"""
        logger.info("="*80)
        logger.info(f"🎬 Starting benchmark: {self.video_path}")
        logger.info("="*80)
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Video: {width}x{height} @ {fps:.1f} FPS, {total_frames_video} frames")
        
        if max_frames:
            logger.info(f"⏱️  Processing max {max_frames} frames")
        
        # Start monitoring
        self.monitor.start()
        
        # Processing loop
        start_time = time.time()
        frame_idx = 0
        
        try:
            while True:
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_idx += 1
                self.total_frames += 1
                
                # Detection
                t0 = time.time()
                detections = self.detector.detect(frame)
                detection_time = time.time() - t0
                self.detection_times.append(detection_time * 1000)  # ms
                
                # Tracking
                t0 = time.time()
                tracks = self.tracker.update(detections, frame.shape[:2])
                tracking_time = time.time() - t0
                self.tracking_times.append(tracking_time * 1000)  # ms
                
                # ReID (only for active tracks)
                # tracks format: [x1, y1, x2, y2, track_id, conf]
                t0 = time.time()
                reid_count = 0

                # Track frame counts (simple simulation)
                if not hasattr(self, 'track_frame_counts'):
                    self.track_frame_counts = {}

                for track in tracks:
                    track_id = int(track[4])

                    # Update frame count for this track
                    if track_id not in self.track_frame_counts:
                        self.track_frame_counts[track_id] = 1
                    else:
                        self.track_frame_counts[track_id] += 1

                    frame_count = self.track_frame_counts[track_id]

                    # Simulate ReID (first 3 frames + re-verify every 30)
                    if frame_count <= 3 or frame_count % 30 == 0:
                        # bbox format: [x1, y1, x2, y2] -> convert to [x, y, w, h]
                        x1, y1, x2, y2 = track[:4]
                        bbox = [x1, y1, x2-x1, y2-y1]
                        embedding = self.extractor.extract(frame, bbox)
                        reid_count += 1

                reid_time = time.time() - t0
                self.reid_times.append(reid_time * 1000)  # ms
                
                # Total frame time
                frame_time = time.time() - frame_start
                self.frame_times.append(frame_time * 1000)  # ms
                
                # Calculate FPS
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                
                # Check frame drop
                if current_fps < fps * 0.8:  # If processing slower than 80% of video FPS
                    self.dropped_frames += 1
                
                # Progress logging
                if frame_idx % 25 == 0:  # Every 25 frames (1 second at 25fps)
                    avg_fps = 1000.0 / np.mean(self.frame_times) if len(self.frame_times) > 0 else 0
                    logger.info(
                        f"Frame {frame_idx:4d} | "
                        f"FPS: {current_fps:5.1f} (avg: {avg_fps:5.1f}) | "
                        f"Det: {detection_time*1000:5.1f}ms | "
                        f"Track: {tracking_time*1000:5.1f}ms | "
                        f"ReID: {reid_time*1000:5.1f}ms ({reid_count} faces) | "
                        f"Tracks: {len(tracks)}"
                    )
                
                # Show video (optional)
                if show_video:
                    cv2.imshow('Benchmark', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Max frames limit
                if max_frames and frame_idx >= max_frames:
                    break
        
        finally:
            cap.release()
            if show_video:
                cv2.destroyAllWindows()
            self.monitor.stop()
        
        # Calculate final statistics
        elapsed_time = time.time() - start_time
        self._print_results(elapsed_time, fps)
    
    def _print_results(self, elapsed_time, video_fps):
        """Print benchmark results"""
        logger.info("="*80)
        logger.info("📊 BENCHMARK RESULTS")
        logger.info("="*80)
        
        # Processing stats
        avg_fps = self.total_frames / elapsed_time
        avg_frame_time = np.mean(self.frame_times)
        
        logger.info(f"\n⏱️  PROCESSING PERFORMANCE:")
        logger.info(f"  Total frames:        {self.total_frames}")
        logger.info(f"  Elapsed time:        {elapsed_time:.2f}s")
        logger.info(f"  Average FPS:         {avg_fps:.2f}")
        logger.info(f"  Video FPS:           {video_fps:.2f}")
        logger.info(f"  Real-time factor:    {avg_fps/video_fps:.2f}x")
        logger.info(f"  Dropped frames:      {self.dropped_frames} ({self.dropped_frames/self.total_frames*100:.1f}%)")
        
        # Component breakdown
        logger.info(f"\n🔍 COMPONENT BREAKDOWN (ms):")
        logger.info(f"  Detection:    avg={np.mean(self.detection_times):6.2f}  max={np.max(self.detection_times):6.2f}  min={np.min(self.detection_times):6.2f}")
        logger.info(f"  Tracking:     avg={np.mean(self.tracking_times):6.2f}  max={np.max(self.tracking_times):6.2f}  min={np.min(self.tracking_times):6.2f}")
        logger.info(f"  ReID:         avg={np.mean(self.reid_times):6.2f}  max={np.max(self.reid_times):6.2f}  min={np.min(self.reid_times):6.2f}")
        logger.info(f"  Total/frame:  avg={avg_frame_time:6.2f}  max={np.max(self.frame_times):6.2f}  min={np.min(self.frame_times):6.2f}")
        
        # Percentage breakdown
        total_avg = avg_frame_time
        det_pct = np.mean(self.detection_times) / total_avg * 100
        track_pct = np.mean(self.tracking_times) / total_avg * 100
        reid_pct = np.mean(self.reid_times) / total_avg * 100
        other_pct = 100 - det_pct - track_pct - reid_pct
        
        logger.info(f"\n📈 TIME DISTRIBUTION:")
        logger.info(f"  Detection:    {det_pct:5.1f}%")
        logger.info(f"  Tracking:     {track_pct:5.1f}%")
        logger.info(f"  ReID:         {reid_pct:5.1f}%")
        logger.info(f"  Other:        {other_pct:5.1f}%")
        
        # System resources
        sys_stats = self.monitor.get_stats()
        if sys_stats:
            logger.info(f"\n💻 SYSTEM RESOURCES:")
            if 'gpu_util_avg' in sys_stats:
                logger.info(f"  GPU Util:     avg={sys_stats['gpu_util_avg']:5.1f}%  max={sys_stats['gpu_util_max']:5.1f}%")
                logger.info(f"  GPU Memory:   avg={sys_stats['gpu_mem_avg']:5.1f}%  max={sys_stats['gpu_mem_max']:5.1f}%")
            logger.info(f"  CPU Util:     avg={sys_stats['cpu_util_avg']:5.1f}%  max={sys_stats['cpu_util_max']:5.1f}%")
            logger.info(f"  RAM Util:     avg={sys_stats['ram_util_avg']:5.1f}%  max={sys_stats['ram_util_max']:5.1f}%")
        
        # Bottleneck analysis
        logger.info(f"\n🎯 BOTTLENECK ANALYSIS:")
        bottleneck = max([
            ('Detection', np.mean(self.detection_times)),
            ('Tracking', np.mean(self.tracking_times)),
            ('ReID', np.mean(self.reid_times))
        ], key=lambda x: x[1])
        logger.info(f"  Primary bottleneck: {bottleneck[0]} ({bottleneck[1]:.2f}ms)")
        
        if avg_fps < video_fps:
            logger.warning(f"\n⚠️  PERFORMANCE WARNING:")
            logger.warning(f"  Processing slower than real-time!")
            logger.warning(f"  Need {video_fps/avg_fps:.2f}x speedup to match video FPS")
        else:
            logger.info(f"\n✅ PERFORMANCE OK:")
            logger.info(f"  Processing faster than real-time!")
        
        logger.info("="*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark detection pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--show", action="store_true", help="Show video while processing")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config file path")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = DetectionBenchmark(args.video, args.config)
    benchmark.run(max_frames=args.max_frames, show_video=args.show)

