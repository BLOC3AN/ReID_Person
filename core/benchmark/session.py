"""Benchmark session manager for resource monitoring."""

import logging
from typing import List, Optional

from .collector import MetricsCollector
from .models import MetricSample, SummaryStats, calculate_summary_stats

logger = logging.getLogger(__name__)


class BenchmarkSession:
    """Manages the lifecycle of a benchmark session as an optional overlay."""

    def __init__(self, detection_stream):
        """Initialize benchmark session.

        Args:
            detection_stream: The detection stream to monitor
        """
        self.detection_stream = detection_stream
        self.is_enabled = False
        self.collector = MetricsCollector(interval_seconds=1.0)

    def enable(self) -> None:
        """Enable benchmark monitoring."""
        if self.is_enabled:
            logger.warning("Benchmark already enabled")
            return

        if not self._is_stream_running():
            raise RuntimeError("Detection stream is not running")

        self.is_enabled = True
        self.collector.start_collection(self.detection_stream)
        logger.info("Benchmark session enabled")

    def disable(self) -> None:
        """Disable benchmark monitoring."""
        if not self.is_enabled:
            logger.warning("Benchmark not enabled")
            return

        self.is_enabled = False
        self.collector.stop_collection()
        logger.info("Benchmark session disabled")

    def is_running(self) -> bool:
        """Check if benchmark is currently running.

        Returns:
            True if benchmark is enabled and collecting metrics
        """
        return self.is_enabled and self.collector.is_collecting

    def get_collected_metrics(self) -> List[MetricSample]:
        """Get all collected metrics.

        Returns:
            List of MetricSample objects
        """
        return self.collector.metrics

    def get_current_metrics(self) -> Optional[MetricSample]:
        """Get the most recent metric sample for real-time display.

        Returns:
            Most recent MetricSample or None if no samples collected
        """
        return self.collector.get_current_metrics()

    def get_summary_statistics(self) -> dict:
        """Calculate and return summary statistics for all metrics.

        Returns:
            Dictionary with metric names as keys and SummaryStats as values
        """
        if not self.collector.metrics:
            raise ValueError("No metrics collected yet")

        metrics = self.collector.metrics

        # Extract values for each metric type
        cpu_values = [m.cpu_percent for m in metrics]
        ram_values = [m.ram_mb for m in metrics]
        gpu_values = [m.gpu_percent for m in metrics]
        vram_values = [m.vram_mb for m in metrics]
        fps_values = [m.fps for m in metrics]
        latency_values = [m.latency_ms for m in metrics]

        return {
            "cpu_percent": calculate_summary_stats("cpu_percent", cpu_values),
            "ram_mb": calculate_summary_stats("ram_mb", ram_values),
            "gpu_percent": calculate_summary_stats("gpu_percent", gpu_values),
            "vram_mb": calculate_summary_stats("vram_mb", vram_values),
            "fps": calculate_summary_stats("fps", fps_values),
            "latency_ms": calculate_summary_stats("latency_ms", latency_values),
        }

    def _is_stream_running(self) -> bool:
        """Check if detection stream is running.

        Returns:
            True if stream is active
        """
        try:
            if self.detection_stream is None:
                return False

            # Check if stream has is_running attribute
            if hasattr(self.detection_stream, "is_running"):
                try:
                    return bool(self.detection_stream.is_running)
                except Exception:
                    pass

            # Check if stream has active attribute
            if hasattr(self.detection_stream, "active"):
                try:
                    return bool(self.detection_stream.active)
                except Exception:
                    pass

            # If stream object exists and no error, assume it's running
            return True

        except Exception as e:
            logger.error(f"Error checking stream status: {e}")
            return False
