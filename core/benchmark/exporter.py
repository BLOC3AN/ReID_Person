"""CSV exporter for benchmark results."""

import csv
import logging
from datetime import datetime
from io import StringIO
from typing import List

from .models import MetricSample, SystemInfo

logger = logging.getLogger(__name__)


class CSVExporter:
    """Exports collected metrics to CSV format."""

    def __init__(self, metrics: List[MetricSample], system_info: SystemInfo):
        """Initialize CSV exporter.

        Args:
            metrics: List of collected MetricSample objects
            system_info: SystemInfo object with system configuration
        """
        self.metrics = metrics
        self.system_info = system_info

    def generate_csv(self) -> str:
        """Generate CSV content as string.

        Returns:
            CSV formatted string with metadata and data rows

        Raises:
            ValueError: If no metrics to export
        """
        if not self.metrics:
            raise ValueError("No metrics to export")

        output = StringIO()

        # Write metadata rows as comments
        output.write(f"# GPU Model: {self.system_info.gpu_model}\n")
        output.write(f"# CPU Model: {self.system_info.cpu_model}\n")
        output.write(f"# Total RAM: {self.system_info.total_ram_gb} GB\n")
        output.write(f"# Total VRAM: {self.system_info.total_vram_gb} GB\n")

        # Write CSV header
        fieldnames = [
            "timestamp",
            "cpu_percent",
            "ram_mb",
            "gpu_percent",
            "vram_mb",
            "fps",
            "latency_ms",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        # Write data rows
        for metric in self.metrics:
            writer.writerow(
                {
                    "timestamp": metric.timestamp.isoformat(),
                    "cpu_percent": f"{metric.cpu_percent:.2f}",
                    "ram_mb": f"{metric.ram_mb:.2f}",
                    "gpu_percent": f"{metric.gpu_percent:.2f}",
                    "vram_mb": f"{metric.vram_mb:.2f}",
                    "fps": f"{metric.fps:.2f}",
                    "latency_ms": f"{metric.latency_ms:.2f}",
                }
            )

        return output.getvalue()

    def save_to_file(self, filepath: str) -> None:
        """Save CSV to file.

        Args:
            filepath: Path where to save the CSV file

        Raises:
            IOError: If unable to write file
        """
        try:
            csv_content = self.generate_csv()
            with open(filepath, "w", newline="") as f:
                f.write(csv_content)
            logger.info(f"CSV exported to {filepath}")
        except IOError as e:
            logger.error(f"Error saving CSV to {filepath}: {e}")
            raise

    def validate_csv_format(self, csv_content: str) -> bool:
        """Validate CSV format.

        Args:
            csv_content: CSV content as string

        Returns:
            True if CSV format is valid

        Raises:
            ValueError: If CSV format is invalid
        """
        try:
            # Normalize line endings
            lines = csv_content.strip().replace("\r\n", "\n").split("\n")

            # Check metadata rows
            if not lines[0].startswith("# GPU Model:"):
                raise ValueError("Missing GPU Model metadata")
            if not lines[1].startswith("# CPU Model:"):
                raise ValueError("Missing CPU Model metadata")
            if not lines[2].startswith("# Total RAM:"):
                raise ValueError("Missing Total RAM metadata")
            if not lines[3].startswith("# Total VRAM:"):
                raise ValueError("Missing Total VRAM metadata")

            # Check header row
            header_line = lines[4].strip()
            expected_columns = [
                "timestamp",
                "cpu_percent",
                "ram_mb",
                "gpu_percent",
                "vram_mb",
                "fps",
                "latency_ms",
            ]
            actual_columns = [col.strip() for col in header_line.split(",")]

            if actual_columns != expected_columns:
                raise ValueError(
                    f"Invalid CSV columns. Expected {expected_columns}, got {actual_columns}"
                )

            # Check data rows
            if len(lines) < 6:
                raise ValueError("No data rows in CSV")

            # Validate first data row format
            reader = csv.DictReader(lines[4:])
            first_row = next(reader, None)
            if not first_row:
                raise ValueError("No data rows in CSV")

            # Check all required fields are present
            for field in expected_columns:
                if field not in first_row:
                    raise ValueError(f"Missing field: {field}")

            return True

        except ValueError:
            raise
        except Exception as e:
            logger.error(f"CSV validation failed: {e}")
            raise ValueError(f"Invalid CSV format: {e}")

    @staticmethod
    def get_system_info() -> SystemInfo:
        """Get current system information.

        Returns:
            SystemInfo object with system configuration
        """
        try:
            import psutil

            # Get CPU model
            cpu_model = "Unknown"
            try:
                import platform

                cpu_model = platform.processor()
            except Exception:
                pass

            # Get RAM info
            ram_info = psutil.virtual_memory()
            total_ram_gb = ram_info.total / (1024**3)

            # Get GPU info
            gpu_model = "No GPU"
            total_vram_gb = 0.0
            try:
                import pynvml

                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    device = pynvml.nvmlDeviceGetHandleByIndex(0)
                    gpu_model = pynvml.nvmlDeviceGetName(device).decode("utf-8")
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(device)
                    total_vram_gb = mem_info.total / (1024**3)
                pynvml.nvmlShutdown()
            except Exception:
                pass

            return SystemInfo(
                gpu_model=gpu_model,
                cpu_model=cpu_model,
                total_ram_gb=total_ram_gb,
                total_vram_gb=total_vram_gb,
            )

        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            # Return default system info
            return SystemInfo(
                gpu_model="Unknown",
                cpu_model="Unknown",
                total_ram_gb=0.0,
                total_vram_gb=0.0,
            )
