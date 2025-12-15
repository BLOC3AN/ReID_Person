"""Data models for benchmark resource monitoring."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class MetricSample:
    """Represents a single metric collection point."""

    timestamp: datetime
    cpu_percent: float
    ram_mb: float
    gpu_percent: float
    vram_mb: float
    fps: float
    latency_ms: float

    def __post_init__(self):
        """Validate metric values are within reasonable ranges."""
        if not 0 <= self.cpu_percent <= 100:
            raise ValueError(f"CPU percent must be 0-100, got {self.cpu_percent}")
        if self.ram_mb < 0:
            raise ValueError(f"RAM MB must be non-negative, got {self.ram_mb}")
        if not 0 <= self.gpu_percent <= 100:
            raise ValueError(f"GPU percent must be 0-100, got {self.gpu_percent}")
        if self.vram_mb < 0:
            raise ValueError(f"VRAM MB must be non-negative, got {self.vram_mb}")
        if self.fps < 0:
            raise ValueError(f"FPS must be non-negative, got {self.fps}")
        if self.latency_ms < 0:
            raise ValueError(f"Latency MS must be non-negative, got {self.latency_ms}")


@dataclass
class SystemInfo:
    """Captures system configuration information."""

    gpu_model: str
    cpu_model: str
    total_ram_gb: float
    total_vram_gb: float

    def __post_init__(self):
        """Validate system info values."""
        if self.total_ram_gb <= 0:
            raise ValueError(f"Total RAM must be positive, got {self.total_ram_gb}")
        if self.total_vram_gb < 0:
            raise ValueError(f"Total VRAM must be non-negative, got {self.total_vram_gb}")


@dataclass
class SummaryStats:
    """Summary statistics for a metric type."""

    metric_name: str
    min_value: float
    max_value: float
    avg_value: float
    sample_count: int

    def __post_init__(self):
        """Validate summary stats."""
        if self.sample_count <= 0:
            raise ValueError(f"Sample count must be positive, got {self.sample_count}")
        if self.min_value > self.max_value:
            raise ValueError(
                f"Min value ({self.min_value}) cannot be greater than max ({self.max_value})"
            )


def calculate_summary_stats(
    metric_name: str, values: List[float]
) -> SummaryStats:
    """Calculate summary statistics from a list of values.

    Args:
        metric_name: Name of the metric
        values: List of metric values

    Returns:
        SummaryStats object with calculated statistics

    Raises:
        ValueError: If values list is empty
    """
    if not values:
        raise ValueError("Cannot calculate stats from empty values list")

    return SummaryStats(
        metric_name=metric_name,
        min_value=min(values),
        max_value=max(values),
        avg_value=sum(values) / len(values),
        sample_count=len(values),
    )
