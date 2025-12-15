"""Metrics collector for system and stream monitoring."""

import logging
import threading
import time
from datetime import datetime
from typing import List, Optional

import psutil

from .models import MetricSample

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects system and stream metrics at regular intervals."""

    def __init__(self, interval_seconds: float = 1.0):
        """Initialize metrics collector.

        Args:
            interval_seconds: Collection interval in seconds (default: 1.0)
        """
        self.interval_seconds = interval_seconds
        self.metrics: List[MetricSample] = []
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Baseline metrics (captured at start)
        self.baseline_cpu_percent = None
        self.baseline_ram_mb = None
        self.baseline_gpu_percent = None
        self.baseline_vram_mb = None

    def collect_sample(self, detection_stream) -> MetricSample:
        """Collect a single metric sample from system and stream.

        Args:
            detection_stream: The detection stream object to extract FPS and latency from

        Returns:
            MetricSample with collected metrics (delta from baseline)

        Raises:
            ValueError: If unable to collect required metrics
        """
        try:
            # Collect current system metrics
            current_cpu_percent = psutil.cpu_percent(interval=0.1)
            ram_info = psutil.virtual_memory()
            current_ram_mb = ram_info.used / (1024 * 1024)

            # Collect GPU metrics
            current_gpu_percent, current_vram_mb = self._get_gpu_metrics()

            # Extract FPS and latency from detection stream (absolute values, not delta)
            fps, latency_ms = self._extract_stream_metrics(detection_stream)

            # Capture baseline on first collection
            if self.baseline_cpu_percent is None:
                self.baseline_cpu_percent = current_cpu_percent
                self.baseline_ram_mb = current_ram_mb
                self.baseline_gpu_percent = current_gpu_percent
                self.baseline_vram_mb = current_vram_mb
                logger.info(f"📊 Baseline captured - CPU: {self.baseline_cpu_percent:.1f}%, RAM: {self.baseline_ram_mb:.0f}MB, GPU: {self.baseline_gpu_percent:.1f}%, VRAM: {self.baseline_vram_mb:.0f}MB")

            # Calculate delta from baseline (resource consumption)
            cpu_percent = max(0, current_cpu_percent - self.baseline_cpu_percent)
            ram_mb = max(0, current_ram_mb - self.baseline_ram_mb)
            gpu_percent = max(0, current_gpu_percent - self.baseline_gpu_percent)
            vram_mb = max(0, current_vram_mb - self.baseline_vram_mb)
            # FPS and Latency are absolute values (AI performance metrics, not resource consumption)

            sample = MetricSample(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                ram_mb=ram_mb,
                gpu_percent=gpu_percent,
                vram_mb=vram_mb,
                fps=fps,
                latency_ms=latency_ms,
            )

            return sample

        except Exception as e:
            logger.error(f"Error collecting metrics sample: {e}")
            raise ValueError(f"Failed to collect metrics: {e}")

    def start_collection(self, detection_stream) -> None:
        """Start continuous metrics collection.

        Args:
            detection_stream: The detection stream to monitor
        """
        if self.is_collecting:
            logger.warning("Metrics collection already running")
            return

        self.is_collecting = True
        self._stop_event.clear()
        self.metrics = []

        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(detection_stream,),
            daemon=True,
        )
        self.collection_thread.start()
        logger.info("Metrics collection started")

    def stop_collection(self) -> List[MetricSample]:
        """Stop metrics collection and return collected samples.

        Returns:
            List of collected MetricSample objects
        """
        if not self.is_collecting:
            logger.warning("Metrics collection not running")
            return self.metrics

        self.is_collecting = False
        self._stop_event.set()

        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)

        logger.info(f"Metrics collection stopped. Collected {len(self.metrics)} samples")
        return self.metrics

    def get_current_metrics(self) -> Optional[MetricSample]:
        """Get the most recent metric sample for real-time display.

        Returns:
            Most recent MetricSample or None if no samples collected
        """
        return self.metrics[-1] if self.metrics else None

    def _collection_loop(self, detection_stream) -> None:
        """Main collection loop running in separate thread.

        Args:
            detection_stream: The detection stream to monitor
        """
        while not self._stop_event.is_set():
            try:
                sample = self.collect_sample(detection_stream)
                self.metrics.append(sample)
                time.sleep(self.interval_seconds)
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                # Continue collecting even if one sample fails
                time.sleep(self.interval_seconds)

    def _get_gpu_metrics(self) -> tuple:
        """Get GPU and VRAM metrics.

        Returns:
            Tuple of (gpu_percent, vram_mb)
        """
        try:
            try:
                import pynvml
            except ImportError:
                logger.warning("pynvml not installed, GPU metrics unavailable")
                return 0.0, 0.0

            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            if device_count == 0:
                return 0.0, 0.0

            # Get metrics from first GPU
            device = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(device)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)

            gpu_percent = float(gpu_util.gpu)
            vram_mb = mem_info.used / (1024 * 1024)

            pynvml.nvmlShutdown()
            return gpu_percent, vram_mb

        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
            return 0.0, 0.0

    def _extract_stream_metrics(self, detection_stream) -> tuple:
        """Extract FPS and latency from detection stream.

        Args:
            detection_stream: The detection stream object

        Returns:
            Tuple of (fps, latency_ms)
        """
        try:
            # Try to extract FPS from stream
            fps = getattr(detection_stream, "fps", 0.0)
            if fps is None:
                fps = 0.0

            # Try to extract latency from stream
            latency_ms = getattr(detection_stream, "latency_ms", 0.0)
            if latency_ms is None:
                latency_ms = 0.0

            return float(fps), float(latency_ms)

        except Exception as e:
            logger.warning(f"Could not extract stream metrics: {e}")
            return 0.0, 0.0
