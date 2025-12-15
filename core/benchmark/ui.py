"""Streamlit UI components for Benchmark Resource feature."""

import streamlit as st
import logging
from typing import Optional, Tuple
from datetime import datetime
import tempfile
import os

from .session import BenchmarkSession
from .exporter import CSVExporter
from .file_manager import get_file_manager

logger = logging.getLogger(__name__)


def render_benchmark_controls(detection_stream) -> Tuple[bool, Optional[int]]:
    """Render benchmark control panel in Streamlit UI.

    Args:
        detection_stream: The detection stream object to monitor

    Returns:
        Tuple of (benchmark_enabled, duration_seconds)
    """
    st.markdown("### 📊 Resource Benchmark")

    col1, col2 = st.columns([1, 2])

    with col1:
        # Benchmark checkbox
        benchmark_enabled = st.checkbox(
            "Enable Benchmark",
            value=False,
            help="Monitor system resource usage (CPU, GPU, RAM, VRAM) during detection",
        )

    with col2:
        if benchmark_enabled:
            # Duration input (only visible when enabled)
            duration_seconds = st.slider(
                "Benchmark Duration (seconds)",
                min_value=10,
                max_value=300,
                value=60,
                step=10,
                help="How long to collect resource metrics",
            )
        else:
            duration_seconds = None

    return benchmark_enabled, duration_seconds


def render_benchmark_metrics_display(benchmark_session: Optional[BenchmarkSession]):
    """Render real-time metrics display during benchmark.

    Args:
        benchmark_session: The active benchmark session
    """
    if not benchmark_session or not benchmark_session.is_running():
        return

    # Get current metrics
    current_metrics = benchmark_session.get_current_metrics()

    if current_metrics:
        # Use a single container to avoid spam
        if "benchmark_metrics_container" not in st.session_state:
            st.session_state.benchmark_metrics_container = st.empty()
        
        with st.session_state.benchmark_metrics_container.container():
            st.markdown("### 📈 Real-time Benchmark Metrics")
            
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "CPU Usage",
                    f"{current_metrics.cpu_percent:.1f}%",
                    delta=None,
                )

            with col2:
                st.metric(
                    "RAM Usage",
                    f"{current_metrics.ram_mb:.0f} MB",
                    delta=None,
                )

            with col3:
                st.metric(
                    "GPU Usage",
                    f"{current_metrics.gpu_percent:.1f}%",
                    delta=None,
                )

            with col4:
                st.metric(
                    "VRAM Usage",
                    f"{current_metrics.vram_mb:.0f} MB",
                    delta=None,
                )

            col5, col6 = st.columns(2)

            with col5:
                st.metric(
                    "FPS",
                    f"{current_metrics.fps:.1f}",
                    delta=None,
                )

            with col6:
                st.metric(
                    "Latency",
                    f"{current_metrics.latency_ms:.1f} ms",
                    delta=None,
                )


def render_benchmark_summary(benchmark_session: Optional[BenchmarkSession]):
    """Render benchmark summary statistics.

    Args:
        benchmark_session: The completed benchmark session
    """
    if not benchmark_session:
        return

    metrics = benchmark_session.get_collected_metrics()
    if not metrics:
        return

    st.markdown("### 📊 Benchmark Summary")

    try:
        stats = benchmark_session.get_summary_statistics()

        # Create summary table
        summary_data = []
        for metric_name, stat in stats.items():
            summary_data.append(
                {
                    "Metric": metric_name.replace("_", " ").title(),
                    "Min": f"{stat.min_value:.2f}",
                    "Max": f"{stat.max_value:.2f}",
                    "Avg": f"{stat.avg_value:.2f}",
                    "Samples": stat.sample_count,
                }
            )

        st.dataframe(summary_data, use_container_width=True)

    except Exception as e:
        logger.error(f"Error rendering benchmark summary: {e}")
        st.error(f"Error calculating summary statistics: {e}")


def render_benchmark_export(benchmark_session: Optional[BenchmarkSession]):
    """Render CSV export button and download functionality.

    Args:
        benchmark_session: The completed benchmark session
    """
    if not benchmark_session:
        return

    metrics = benchmark_session.get_collected_metrics()
    if not metrics:
        return

    st.markdown("### 💾 Export Results")

    try:
        # Get system info
        system_info = CSVExporter.get_system_info()

        # Create exporter
        exporter = CSVExporter(metrics, system_info)

        # Generate CSV
        csv_content = exporter.generate_csv()

        # Validate CSV format
        exporter.validate_csv_format(csv_content)

        # Auto-save to outputs/benchmark/
        import os
        from pathlib import Path
        
        benchmark_dir = Path("outputs/benchmark")
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.csv"
        filepath = benchmark_dir / filename
        
        with open(filepath, "w") as f:
            f.write(csv_content)
        
        logger.info(f"✅ Benchmark CSV auto-saved to {filepath}")

        # Download button
        st.download_button(
            label="📥 Download Benchmark Results (CSV)",
            data=csv_content,
            file_name=filename,
            mime="text/csv",
            key=f"download_csv_{timestamp}",
        )

        st.success(f"✅ CSV saved to {filepath} ({len(metrics)} samples)")

    except Exception as e:
        logger.error(f"Error exporting benchmark results: {e}")
        st.error(f"Error exporting results: {e}")


def initialize_benchmark_session(detection_stream) -> Optional[BenchmarkSession]:
    """Initialize benchmark session in Streamlit session state.

    Args:
        detection_stream: The detection stream to monitor

    Returns:
        BenchmarkSession object or None if stream not running
    """
    if "benchmark_session" not in st.session_state:
        try:
            session = BenchmarkSession(detection_stream)
            st.session_state.benchmark_session = session
            return session
        except Exception as e:
            logger.error(f"Error initializing benchmark session: {e}")
            return None

    return st.session_state.benchmark_session


def cleanup_benchmark_session():
    """Clean up benchmark session from Streamlit session state."""
    if "benchmark_session" in st.session_state:
        session = st.session_state.benchmark_session
        if session and session.is_running():
            session.disable()
        del st.session_state.benchmark_session
