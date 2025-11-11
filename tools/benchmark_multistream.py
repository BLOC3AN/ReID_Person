#!/usr/bin/env python3
"""
Multi-Stream Benchmark for Triton Inference Server
Simulates concurrent camera streams to measure throughput and latency
"""

import argparse
import time
import numpy as np
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from loguru import logger
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import YOLOXDetector
from core.detector_trt import TensorRTDetector
from core.detector_triton import TritonDetector


class MultiStreamBenchmark:
    """Benchmark detector with multiple concurrent streams"""
    
    def __init__(
        self,
        detector,
        num_streams: int = 4,
        num_iterations: int = 100,
        warmup: int = 10
    ):
        self.detector = detector
        self.num_streams = num_streams
        self.num_iterations = num_iterations
        self.warmup = warmup
        
        # Generate dummy frames
        self.frames = self._generate_frames()
    
    def _generate_frames(self) -> List[np.ndarray]:
        """Generate random test frames"""
        logger.info(f"Generating {self.num_streams} test frames...")
        frames = []
        for i in range(self.num_streams):
            # Random image with some patterns
            frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
            # Add some rectangles to simulate objects
            for _ in range(5):
                x1, y1 = np.random.randint(0, 1000, 2)
                x2, y2 = x1 + np.random.randint(50, 200), y1 + np.random.randint(50, 200)
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            frames.append(frame)
        return frames
    
    def _process_stream(self, stream_id: int, iteration: int) -> Tuple[int, int, float, int]:
        """Process single frame from a stream"""
        frame = self.frames[stream_id % len(self.frames)]
        
        start_time = time.time()
        detections = self.detector.detect(frame)
        latency = time.time() - start_time
        
        return stream_id, iteration, latency, len(detections)
    
    def run_sequential(self) -> dict:
        """Run sequential inference (baseline)"""
        logger.info(f"Running sequential benchmark ({self.num_iterations} iterations)...")
        
        latencies = []
        
        # Warmup
        for _ in range(self.warmup):
            self.detector.detect(self.frames[0])
        
        # Benchmark
        start_time = time.time()
        for i in range(self.num_iterations):
            for stream_id in range(self.num_streams):
                _, _, latency, _ = self._process_stream(stream_id, i)
                latencies.append(latency)
        
        total_time = time.time() - start_time
        
        return {
            'mode': 'sequential',
            'total_requests': len(latencies),
            'total_time': total_time,
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'throughput': len(latencies) / total_time
        }
    
    def run_concurrent(self) -> dict:
        """Run concurrent inference (multi-stream)"""
        logger.info(f"Running concurrent benchmark ({self.num_streams} streams, {self.num_iterations} iterations)...")
        
        latencies = []
        
        # Warmup
        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
            futures = [executor.submit(self._process_stream, i, 0) for i in range(self.warmup)]
            for future in as_completed(futures):
                future.result()
        
        # Benchmark
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.num_streams) as executor:
            futures = []
            for i in range(self.num_iterations):
                for stream_id in range(self.num_streams):
                    future = executor.submit(self._process_stream, stream_id, i)
                    futures.append(future)
            
            for future in as_completed(futures):
                stream_id, iteration, latency, num_dets = future.result()
                latencies.append(latency)
        
        total_time = time.time() - start_time
        
        return {
            'mode': 'concurrent',
            'total_requests': len(latencies),
            'total_time': total_time,
            'avg_latency': np.mean(latencies),
            'p50_latency': np.percentile(latencies, 50),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99),
            'throughput': len(latencies) / total_time
        }
    
    def print_results(self, results: dict):
        """Print benchmark results"""
        print(f"\n{'='*60}")
        print(f"Mode: {results['mode'].upper()}")
        print(f"{'='*60}")
        print(f"Total Requests:    {results['total_requests']}")
        print(f"Total Time:        {results['total_time']:.2f}s")
        print(f"Throughput:        {results['throughput']:.2f} FPS")
        print(f"Avg Latency:       {results['avg_latency']*1000:.2f}ms")
        print(f"P50 Latency:       {results['p50_latency']*1000:.2f}ms")
        print(f"P95 Latency:       {results['p95_latency']*1000:.2f}ms")
        print(f"P99 Latency:       {results['p99_latency']*1000:.2f}ms")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Multi-Stream Benchmark')
    parser.add_argument('--backend', type=str, default='triton',
                        choices=['pytorch', 'tensorrt', 'triton'],
                        help='Detector backend')
    parser.add_argument('--pytorch-model', type=str, default='models/bytetrack_x_mot17.pth.tar',
                        help='PyTorch model path')
    parser.add_argument('--tensorrt-engine', type=str, default='models/bytetrack_x_mot17_fp16.trt',
                        help='TensorRT engine path')
    parser.add_argument('--triton-url', type=str, default='localhost:8001',
                        help='Triton server URL')
    parser.add_argument('--triton-model', type=str, default='bytetrack_tensorrt',
                        help='Triton model name')
    parser.add_argument('--num-streams', type=int, default=4,
                        help='Number of concurrent streams')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of iterations per stream')
    parser.add_argument('--warmup', type=int, default=10,
                        help='Number of warmup iterations')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.45,
                        help='NMS threshold')
    
    args = parser.parse_args()
    
    # Initialize detector
    logger.info(f"Initializing {args.backend} detector...")
    
    if args.backend == 'pytorch':
        detector = YOLOXDetector(
            model_path=args.pytorch_model,
            model_type='mot17',
            device='cuda',
            fp16=True,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            test_size=(640, 640)
        )
    elif args.backend == 'tensorrt':
        detector = TensorRTDetector(
            engine_path=args.tensorrt_engine,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            test_size=(640, 640)
        )
    else:  # triton
        detector = TritonDetector(
            triton_url=args.triton_url,
            model_name=args.triton_model,
            conf_thresh=args.conf_thresh,
            nms_thresh=args.nms_thresh,
            test_size=(640, 640)
        )
    
    logger.info(f"Detector initialized: {args.backend}")
    
    # Run benchmark
    benchmark = MultiStreamBenchmark(
        detector=detector,
        num_streams=args.num_streams,
        num_iterations=args.iterations,
        warmup=args.warmup
    )
    
    # Sequential benchmark
    seq_results = benchmark.run_sequential()
    benchmark.print_results(seq_results)
    
    # Concurrent benchmark
    conc_results = benchmark.run_concurrent()
    benchmark.print_results(conc_results)
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Sequential Throughput:  {seq_results['throughput']:.2f} FPS")
    print(f"Concurrent Throughput:  {conc_results['throughput']:.2f} FPS")
    print(f"Speedup:                {conc_results['throughput']/seq_results['throughput']:.2f}x")
    print(f"")
    print(f"Sequential Avg Latency: {seq_results['avg_latency']*1000:.2f}ms")
    print(f"Concurrent Avg Latency: {conc_results['avg_latency']*1000:.2f}ms")
    print(f"Latency Overhead:       {(conc_results['avg_latency']/seq_results['avg_latency']-1)*100:.1f}%")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()

