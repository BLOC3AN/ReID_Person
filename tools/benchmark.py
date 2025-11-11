#!/usr/bin/env python3
"""
Benchmark PyTorch vs TensorRT detector
Compare speed and accuracy
"""

import sys
import time
import argparse
import numpy as np
import cv2
from pathlib import Path
from loguru import logger
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import YOLOXDetector
from core.detector_trt import TensorRTDetector


def benchmark_detector(detector, frames: List[np.ndarray], warmup: int = 10, iterations: int = 100):
    """
    Benchmark detector performance
    
    Args:
        detector: Detector instance
        frames: List of test frames
        warmup: Number of warmup iterations
        iterations: Number of benchmark iterations
        
    Returns:
        results: Dict with timing statistics
    """
    logger.info(f"\n🔥 Warming up ({warmup} iterations)...")
    for i in range(warmup):
        frame = frames[i % len(frames)]
        _ = detector.detect(frame)
    
    logger.info(f"⏱️  Benchmarking ({iterations} iterations)...")
    times = []
    all_detections = []
    
    for i in range(iterations):
        frame = frames[i % len(frames)]
        
        start = time.time()
        detections = detector.detect(frame)
        elapsed = (time.time() - start) * 1000  # ms
        
        times.append(elapsed)
        all_detections.append(detections)
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    p50_time = np.percentile(times, 50)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)
    fps = 1000 / avg_time
    
    results = {
        'avg_time': avg_time,
        'std_time': std_time,
        'min_time': min_time,
        'max_time': max_time,
        'p50_time': p50_time,
        'p95_time': p95_time,
        'p99_time': p99_time,
        'fps': fps,
        'detections': all_detections
    }
    
    return results


def compare_detections(det1: np.ndarray, det2: np.ndarray, iou_thresh: float = 0.5):
    """
    Compare two detection results
    
    Args:
        det1: First detection (N1, 7)
        det2: Second detection (N2, 7)
        iou_thresh: IoU threshold for matching
        
    Returns:
        metrics: Dict with comparison metrics
    """
    if len(det1) == 0 and len(det2) == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'matched': 0}
    
    if len(det1) == 0 or len(det2) == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'matched': 0}
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(det1), len(det2)))
    
    for i, box1 in enumerate(det1[:, :4]):
        for j, box2 in enumerate(det2[:, :4]):
            iou_matrix[i, j] = calculate_iou(box1, box2)
    
    # Match detections
    matched = 0
    for i in range(len(det1)):
        if np.max(iou_matrix[i]) >= iou_thresh:
            matched += 1
    
    precision = matched / len(det2) if len(det2) > 0 else 0
    recall = matched / len(det1) if len(det1) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'total_det1': len(det1),
        'total_det2': len(det2)
    }


def calculate_iou(box1, box2):
    """Calculate IoU between two boxes"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Union
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def load_test_frames(video_path: str, num_frames: int = 100) -> List[np.ndarray]:
    """Load test frames from video"""
    logger.info(f"📹 Loading test frames from {video_path}...")
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)
    
    frame_idx = 0
    while len(frames) < num_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frames.append(frame)
        frame_idx += step
    
    cap.release()
    
    logger.info(f"✅ Loaded {len(frames)} frames")
    return frames


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch vs TensorRT")
    parser.add_argument(
        "--pytorch-model",
        default="models/bytetrack_x_mot17.pth.tar",
        help="PyTorch model path"
    )
    parser.add_argument(
        "--tensorrt-engine",
        default="models/bytetrack_x_mot17_fp16.trt",
        help="TensorRT engine path"
    )
    parser.add_argument(
        "--video",
        default="data/test_video.mp4",
        help="Test video path"
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=100,
        help="Number of test frames"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Benchmark iterations"
    )
    parser.add_argument(
        "--skip-pytorch",
        action="store_true",
        help="Skip PyTorch benchmark"
    )
    parser.add_argument(
        "--skip-tensorrt",
        action="store_true",
        help="Skip TensorRT benchmark"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("BENCHMARK: PYTORCH VS TENSORRT")
    logger.info("=" * 80)
    
    # Load test frames
    frames = load_test_frames(args.video, args.num_frames)
    
    if len(frames) == 0:
        logger.error("❌ No frames loaded")
        return
    
    results = {}
    
    # Benchmark PyTorch
    if not args.skip_pytorch:
        logger.info("\n" + "=" * 80)
        logger.info("🔥 PYTORCH DETECTOR")
        logger.info("=" * 80)
        
        try:
            pytorch_detector = YOLOXDetector(
                model_path=args.pytorch_model,
                model_type='mot17',
                device='cuda',
                fp16=True,
                conf_thresh=0.5,
                nms_thresh=0.45,
                test_size=(640, 640)
            )
            
            results['pytorch'] = benchmark_detector(
                pytorch_detector, frames, args.warmup, args.iterations
            )
            
            logger.info("\n📊 PyTorch Results:")
            logger.info(f"  Average: {results['pytorch']['avg_time']:.2f} ± {results['pytorch']['std_time']:.2f} ms")
            logger.info(f"  Min: {results['pytorch']['min_time']:.2f} ms")
            logger.info(f"  Max: {results['pytorch']['max_time']:.2f} ms")
            logger.info(f"  P50: {results['pytorch']['p50_time']:.2f} ms")
            logger.info(f"  P95: {results['pytorch']['p95_time']:.2f} ms")
            logger.info(f"  P99: {results['pytorch']['p99_time']:.2f} ms")
            logger.info(f"  FPS: {results['pytorch']['fps']:.2f}")
            
        except Exception as e:
            logger.error(f"❌ PyTorch benchmark failed: {e}")
    
    # Benchmark TensorRT
    if not args.skip_tensorrt:
        logger.info("\n" + "=" * 80)
        logger.info("⚡ TENSORRT DETECTOR")
        logger.info("=" * 80)
        
        if not Path(args.tensorrt_engine).exists():
            logger.error(f"❌ TensorRT engine not found: {args.tensorrt_engine}")
            logger.info("💡 Convert ONNX to TensorRT first:")
            logger.info(f"   python tools/convert_tensorrt.py --onnx <onnx_path>")
        else:
            try:
                tensorrt_detector = TensorRTDetector(
                    engine_path=args.tensorrt_engine,
                    conf_thresh=0.5,
                    nms_thresh=0.45,
                    test_size=(640, 640)
                )
                
                results['tensorrt'] = benchmark_detector(
                    tensorrt_detector, frames, args.warmup, args.iterations
                )
                
                logger.info("\n📊 TensorRT Results:")
                logger.info(f"  Average: {results['tensorrt']['avg_time']:.2f} ± {results['tensorrt']['std_time']:.2f} ms")
                logger.info(f"  Min: {results['tensorrt']['min_time']:.2f} ms")
                logger.info(f"  Max: {results['tensorrt']['max_time']:.2f} ms")
                logger.info(f"  P50: {results['tensorrt']['p50_time']:.2f} ms")
                logger.info(f"  P95: {results['tensorrt']['p95_time']:.2f} ms")
                logger.info(f"  P99: {results['tensorrt']['p99_time']:.2f} ms")
                logger.info(f"  FPS: {results['tensorrt']['fps']:.2f}")
                
            except Exception as e:
                logger.error(f"❌ TensorRT benchmark failed: {e}")
    
    # Compare results
    if 'pytorch' in results and 'tensorrt' in results:
        logger.info("\n" + "=" * 80)
        logger.info("📊 COMPARISON")
        logger.info("=" * 80)
        
        speedup = results['pytorch']['avg_time'] / results['tensorrt']['avg_time']
        fps_gain = results['tensorrt']['fps'] - results['pytorch']['fps']
        
        logger.info(f"\n⚡ Speed:")
        logger.info(f"  PyTorch: {results['pytorch']['avg_time']:.2f} ms ({results['pytorch']['fps']:.2f} FPS)")
        logger.info(f"  TensorRT: {results['tensorrt']['avg_time']:.2f} ms ({results['tensorrt']['fps']:.2f} FPS)")
        logger.info(f"  Speedup: {speedup:.2f}x")
        logger.info(f"  FPS Gain: +{fps_gain:.2f} FPS")
        
        # Compare accuracy
        logger.info(f"\n🎯 Accuracy (comparing {len(frames)} frames):")
        
        all_metrics = []
        for i in range(len(frames)):
            pytorch_det = results['pytorch']['detections'][i]
            tensorrt_det = results['tensorrt']['detections'][i]
            metrics = compare_detections(pytorch_det, tensorrt_det)
            all_metrics.append(metrics)
        
        avg_precision = np.mean([m['precision'] for m in all_metrics])
        avg_recall = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        
        logger.info(f"  Precision: {avg_precision:.4f}")
        logger.info(f"  Recall: {avg_recall:.4f}")
        logger.info(f"  F1 Score: {avg_f1:.4f}")
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ BENCHMARK COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

