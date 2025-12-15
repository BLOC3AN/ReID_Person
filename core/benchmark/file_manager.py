"""File management utilities for benchmark CSV exports."""

import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class BenchmarkFileManager:
    """Manages benchmark CSV file generation and cleanup."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize file manager.

        Args:
            output_dir: Directory for CSV files. If None, uses system temp directory.
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = Path(tempfile.gettempdir()) / "benchmark_results"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Benchmark file manager initialized: {self.output_dir}")

    def generate_filename(self, prefix: str = "benchmark") -> str:
        """Generate unique filename with timestamp.

        Args:
            prefix: Filename prefix (default: "benchmark")

        Returns:
            Filename with timestamp (e.g., "benchmark_20240115_103000.csv")
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.csv"

    def get_filepath(self, filename: str) -> Path:
        """Get full filepath for a filename.

        Args:
            filename: Filename

        Returns:
            Full path to file
        """
        return self.output_dir / filename

    def save_csv(self, csv_content: str, filename: Optional[str] = None) -> str:
        """Save CSV content to file.

        Args:
            csv_content: CSV content as string
            filename: Optional filename. If None, generates one with timestamp.

        Returns:
            Full filepath where file was saved

        Raises:
            IOError: If unable to write file
        """
        if not filename:
            filename = self.generate_filename()

        filepath = self.get_filepath(filename)

        try:
            with open(filepath, "w", newline="") as f:
                f.write(csv_content)

            logger.info(f"CSV saved to {filepath}")
            return str(filepath)

        except IOError as e:
            logger.error(f"Error saving CSV to {filepath}: {e}")
            raise

    def cleanup_old_files(self, max_age_hours: int = 24) -> int:
        """Clean up old benchmark CSV files.

        Args:
            max_age_hours: Maximum age of files to keep (default: 24 hours)

        Returns:
            Number of files deleted
        """
        import time

        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_count = 0

        try:
            for filepath in self.output_dir.glob("benchmark_*.csv"):
                file_age = current_time - filepath.stat().st_mtime

                if file_age > max_age_seconds:
                    filepath.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old benchmark file: {filepath}")

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} old benchmark files")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up old files: {e}")
            return 0

    def list_benchmark_files(self) -> list:
        """List all benchmark CSV files.

        Returns:
            List of benchmark CSV filepaths
        """
        try:
            files = sorted(self.output_dir.glob("benchmark_*.csv"), reverse=True)
            return [str(f) for f in files]
        except Exception as e:
            logger.error(f"Error listing benchmark files: {e}")
            return []

    def get_file_size(self, filepath: str) -> int:
        """Get file size in bytes.

        Args:
            filepath: Path to file

        Returns:
            File size in bytes, or 0 if file doesn't exist
        """
        try:
            return os.path.getsize(filepath)
        except OSError:
            return 0

    def delete_file(self, filepath: str) -> bool:
        """Delete a benchmark file.

        Args:
            filepath: Path to file to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            Path(filepath).unlink()
            logger.info(f"Deleted benchmark file: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error deleting file {filepath}: {e}")
            return False


# Global file manager instance
_file_manager: Optional[BenchmarkFileManager] = None


def get_file_manager(output_dir: Optional[str] = None) -> BenchmarkFileManager:
    """Get or create global file manager instance.

    Args:
        output_dir: Optional output directory for CSV files

    Returns:
        BenchmarkFileManager instance
    """
    global _file_manager

    if _file_manager is None:
        _file_manager = BenchmarkFileManager(output_dir)

    return _file_manager
