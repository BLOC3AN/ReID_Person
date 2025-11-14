#!/usr/bin/env python3
"""
Multi-Stream Benchmark - Test real-time performance with multiple camera streams
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

sys.path.insert(0, str(Path(__file__).parent))

from core.detector_triton import TritonDetector
from core.tracker import ByteTrackWrapper
from core.arcface_triton_client import ArcFaceTritonClient


class StreamStats:
    """Statistics for a single stream"""
    
    def __init__(self, stream_id):
        self.stream_id = stream_id
        self.frame_count = 0
        self.dropped_frames = 0
        self.frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.tracking_times = deque(maxlen=100)
        self.reid_times = deque(maxlen=100)
        self.track_frame_counts = {}
        self.start_time = time.time()
    
    def get_fps(self):
        if len(self.frame_times) == 0:
            return 0.0
        return 1000.0 / np.mean(self.frame_times)
    
    def get_total_fps(self):
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.frame_count / elapsed


class MultiStreamBenchmark:
    """Benchmark with multiple camera streams"""
    
    def __init__(self, stream_urls, config_path=".streamlit/configs/config.yaml"):
        self.stream_urls = stream_urls
        self.num_streams = len(stream_urls)
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize shared components
        logger.info(f"🔧 Initializing components for {self.num_streams} streams...")
        self._init_components()
        
        # Per-stream stats
        self.stream_stats = [StreamStats(i) for i in range(self.num_streams)]
        
        # System monitor
        self.monitor = PerformanceMonitor(gpu_id=0)
        
        # Control
        self.running = False
    
    def _init_components(self):
        """Initialize detection components (shared across streams)"""
        cfg = self.config['detection']
        
        # Detector (Triton) - shared
        if cfg['backend'] == 'triton':
            logger.info("📡 Using Triton Inference Server (shared)")
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
            raise ValueError(f"Only Triton backend supported")
        
        # Tracker - one per stream (ByteTrack)
        track_cfg = self.config['tracking']
        self.trackers = []

        from core.tracker import ByteTrackWrapper
        for i in range(self.num_streams):
            tracker = ByteTrackWrapper(
                track_thresh=track_cfg.get('track_thresh', 0.5),
                track_buffer=track_cfg.get('track_buffer', 30),
                match_thresh=track_cfg.get('match_thresh', 0.8),
                frame_rate=30,
                mot20=track_cfg.get('mot20', False)
            )
            self.trackers.append(tracker)
        
        # Feature extractor (ArcFace Triton) - shared
        triton_url = self.config['detection']['triton']['url']
        self.extractor = ArcFaceTritonClient(
            triton_url=triton_url,
            model_name='arcface_tensorrt'
        )
        
        logger.info("✅ All components initialized")
    
    def _process_stream(self, stream_id, stream_url, max_frames=None):
        """Process a single stream (runs in thread)"""
        stats = self.stream_stats[stream_id]
        tracker = self.trackers[stream_id]
        
        logger.info(f"[Stream {stream_id}] 📹 Opening: {stream_url}")
        
        # Open stream
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            logger.error(f"[Stream {stream_id}] ❌ Cannot open stream")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0:
            fps = 25.0  # Default for RTSP
        
        logger.info(f"[Stream {stream_id}] ✅ Stream opened @ {fps:.1f} FPS")
        
        try:
            while self.running:
                frame_start = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"[Stream {stream_id}] Failed to read frame, reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = cv2.VideoCapture(stream_url)
                    continue
                
                stats.frame_count += 1
                
                # Detection (shared detector)
                t0 = time.time()
                detections = self.detector.detect(frame)
                detection_time = time.time() - t0
                stats.detection_times.append(detection_time * 1000)
                
                # Tracking (per-stream tracker)
                t0 = time.time()
                tracks = tracker.update(detections, frame.shape[:2])
                tracking_time = time.time() - t0
                stats.tracking_times.append(tracking_time * 1000)
                
                # ReID (shared extractor, but only for new/periodic tracks)
                t0 = time.time()
                reid_count = 0
                for track in tracks:
                    track_id = int(track[4])
                    
                    if track_id not in stats.track_frame_counts:
                        stats.track_frame_counts[track_id] = 1
                    else:
                        stats.track_frame_counts[track_id] += 1
                    
                    frame_count = stats.track_frame_counts[track_id]
                    
                    if frame_count <= 3 or frame_count % 30 == 0:
                        x1, y1, x2, y2 = track[:4]
                        bbox = [x1, y1, x2-x1, y2-y1]
                        embedding = self.extractor.extract(frame, bbox)
                        reid_count += 1
                
                reid_time = time.time() - t0
                stats.reid_times.append(reid_time * 1000)
                
                # Total frame time
                frame_time = time.time() - frame_start
                stats.frame_times.append(frame_time * 1000)
                
                # Check frame drop
                current_fps = 1.0 / frame_time if frame_time > 0 else 0
                if current_fps < fps * 0.8:
                    stats.dropped_frames += 1
                
                # Max frames limit
                if max_frames and stats.frame_count >= max_frames:
                    break
        
        finally:
            cap.release()
            logger.info(f"[Stream {stream_id}] 🛑 Stopped")
    
    def run(self, duration_seconds=60, max_frames_per_stream=None):
        """Run benchmark with all streams"""
        logger.info("="*80)
        logger.info(f"🎬 Starting multi-stream benchmark")
        logger.info(f"   Streams: {self.num_streams}")
        logger.info(f"   Duration: {duration_seconds}s")
        logger.info("="*80)
        
        # Start monitoring
        self.monitor.start()
        
        # Start stream threads
        self.running = True
        threads = []
        
        for i, url in enumerate(self.stream_urls):
            thread = threading.Thread(
                target=self._process_stream,
                args=(i, url, max_frames_per_stream),
                daemon=True
            )
            thread.start()
            threads.append(thread)
            time.sleep(0.5)  # Stagger starts
        
        # Monitor progress
        start_time = time.time()
        try:
            while self.running:
                time.sleep(5)  # Report every 5 seconds
                
                elapsed = time.time() - start_time
                
                # Print stats
                logger.info("─" * 80)
                logger.info(f"⏱️  Elapsed: {elapsed:.1f}s")
                
                total_frames = 0
                total_fps = 0
                
                for i, stats in enumerate(self.stream_stats):
                    fps = stats.get_fps()
                    total_fps_stream = stats.get_total_fps()
                    total_frames += stats.frame_count
                    
                    det_avg = np.mean(stats.detection_times) if len(stats.detection_times) > 0 else 0
                    track_avg = np.mean(stats.tracking_times) if len(stats.tracking_times) > 0 else 0
                    reid_avg = np.mean(stats.reid_times) if len(stats.reid_times) > 0 else 0
                    
                    logger.info(
                        f"  Stream {i}: {stats.frame_count:4d} frames | "
                        f"FPS: {fps:5.1f} (avg: {total_fps_stream:5.1f}) | "
                        f"Det: {det_avg:5.1f}ms | Track: {track_avg:4.1f}ms | ReID: {reid_avg:4.1f}ms | "
                        f"Drops: {stats.dropped_frames}"
                    )
                    total_fps += total_fps_stream
                
                logger.info(f"  TOTAL: {total_frames} frames | Combined FPS: {total_fps:.1f}")
                
                # Check duration
                if elapsed >= duration_seconds:
                    logger.info(f"⏰ Duration reached ({duration_seconds}s)")
                    break
        
        except KeyboardInterrupt:
            logger.info("⚠️  Interrupted by user")
        
        finally:
            # Stop all streams
            self.running = False
            
            # Wait for threads
            for thread in threads:
                thread.join(timeout=2)
            
            # Stop monitoring
            self.monitor.stop()
        
        # Print final results
        self._print_results(time.time() - start_time)
    
    def _print_results(self, elapsed_time):
        """Print final benchmark results"""
        logger.info("="*80)
        logger.info("📊 MULTI-STREAM BENCHMARK RESULTS")
        logger.info("="*80)
        
        total_frames = sum(s.frame_count for s in self.stream_stats)
        total_dropped = sum(s.dropped_frames for s in self.stream_stats)
        
        logger.info(f"\n⏱️  OVERALL PERFORMANCE:")
        logger.info(f"  Duration:            {elapsed_time:.2f}s")
        logger.info(f"  Total frames:        {total_frames}")
        logger.info(f"  Combined FPS:        {total_frames/elapsed_time:.2f}")
        logger.info(f"  Dropped frames:      {total_dropped} ({total_dropped/total_frames*100:.1f}%)")
        
        logger.info(f"\n📹 PER-STREAM BREAKDOWN:")
        for i, stats in enumerate(self.stream_stats):
            logger.info(f"\n  Stream {i}:")
            logger.info(f"    Frames:          {stats.frame_count}")
            logger.info(f"    FPS:             {stats.get_total_fps():.2f}")
            logger.info(f"    Dropped:         {stats.dropped_frames} ({stats.dropped_frames/max(stats.frame_count,1)*100:.1f}%)")
            
            if len(stats.detection_times) > 0:
                logger.info(f"    Detection:       avg={np.mean(stats.detection_times):6.2f}ms  max={np.max(stats.detection_times):6.2f}ms")
                logger.info(f"    Tracking:        avg={np.mean(stats.tracking_times):6.2f}ms  max={np.max(stats.tracking_times):6.2f}ms")
                logger.info(f"    ReID:            avg={np.mean(stats.reid_times):6.2f}ms  max={np.max(stats.reid_times):6.2f}ms")
        
        # System resources
        sys_stats = self.monitor.get_stats()
        if sys_stats:
            logger.info(f"\n💻 SYSTEM RESOURCES:")
            if 'gpu_util_avg' in sys_stats:
                logger.info(f"  GPU Util:        avg={sys_stats['gpu_util_avg']:5.1f}%  max={sys_stats['gpu_util_max']:5.1f}%")
                logger.info(f"  GPU Memory:      avg={sys_stats['gpu_mem_avg']:5.1f}%  max={sys_stats['gpu_mem_max']:5.1f}%")
            logger.info(f"  CPU Util:        avg={sys_stats['cpu_util_avg']:5.1f}%  max={sys_stats['cpu_util_max']:5.1f}%")
            logger.info(f"  RAM Util:        avg={sys_stats['ram_util_avg']:5.1f}%  max={sys_stats['ram_util_max']:5.1f}%")
        
        logger.info("="*80)


class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.running = False
        self.metrics = {
            'gpu_util': deque(maxlen=200),
            'gpu_mem': deque(maxlen=200),
            'cpu_util': deque(maxlen=200),
            'ram_util': deque(maxlen=200),
        }
        
        if GPU_AVAILABLE:
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
    
    def _monitor_loop(self):
        while self.running:
            self.metrics['cpu_util'].append(psutil.cpu_percent(interval=0.1))
            self.metrics['ram_util'].append(psutil.virtual_memory().percent)
            
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
        stats = {}
        for key, values in self.metrics.items():
            if len(values) > 0:
                stats[f'{key}_avg'] = np.mean(values)
                stats[f'{key}_max'] = np.max(values)
                stats[f'{key}_min'] = np.min(values)
        return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-stream benchmark")
    parser.add_argument("--streams", type=str, nargs='+', required=True, help="Stream URLs")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames per stream")
    parser.add_argument("--config", type=str, default=".streamlit/configs/config.yaml", help="Config file")
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = MultiStreamBenchmark(args.streams, args.config)
    benchmark.run(duration_seconds=args.duration, max_frames_per_stream=args.max_frames)

