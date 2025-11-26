#!/usr/bin/env python3
"""
GStreamer-based Stream Writer for non-blocking livestream output.

This module provides a GStreamer pipeline with appsrc for realtime HLS streaming
without blocking the main processing thread.
"""

from pathlib import Path
from typing import Optional
import numpy as np
from loguru import logger
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib


# Initialize GStreamer
Gst.init(None)


class GStreamerWriter:
    """
    Non-blocking GStreamer-based livestream writer.
    Uses appsrc with leaky queue for realtime streaming without blocking.

    Pipeline: appsrc → videoconvert → nvh264enc → hlssink
    """

    def __init__(self, output_dir: str, width: int, height: int, fps: float,
                 hls_segment_time: int = 6, hls_list_size: int = 10, max_files: int = 10):
        """
        Initialize GStreamer pipeline for HLS livestream.

        Args:
            output_dir: Directory to save HLS files
            width: Frame width
            height: Frame height
            fps: Frames per second
            hls_segment_time: HLS segment duration in seconds
            hls_list_size: Number of segments in playlist
            max_files: Maximum number of segments to keep on disk (old segments auto-deleted)
        """
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.fps = fps
        self.hls_segment_time = hls_segment_time
        self.hls_list_size = hls_list_size
        self.max_files = max_files

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hls_playlist = self.output_dir / "stream.m3u8"

        # Frame counters
        self.frames_sent = 0
        self.frames_dropped = 0

        # Build and start GStreamer pipeline
        self._build_pipeline()

        logger.info(f"🎬 GStreamer Writer initialized")
        logger.info(f"   Output: {self.hls_playlist}")
        logger.info(f"   Resolution: {width}x{height} @ {fps} fps")
        logger.info(f"   HLS: {hls_segment_time}s segments, {hls_list_size} playlist size, max {max_files} files on disk")

    def _build_pipeline(self):
        """Build GStreamer pipeline with appsrc → nvh264enc → hlssink"""

        # Calculate frame size
        frame_size = self.width * self.height * 3  # BGR24

        # Build pipeline string
        # Note: Add format=yuv420p to ensure browser compatibility (avoid yuvj420p)
        # mpegtsmux requires byte-stream format, not avc
        pipeline_str = (
            f"appsrc name=source is-live=true format=time "
            f"caps=video/x-raw,format=BGR,width={self.width},height={self.height},"
            f"framerate={int(self.fps)}/1 "
            f"max-bytes=0 block=false leaky-type=downstream "
            f"! videoconvert "
            f"! video/x-raw,format=I420 "
            f"! nvh264enc preset=low-latency-hp rc-mode=cbr bitrate=4000 gop-size={int(self.fps * self.hls_segment_time)} "
            f"! video/x-h264,stream-format=byte-stream,alignment=au "
            f"! h264parse config-interval=-1 "
            f"! mpegtsmux "
            f"! hlssink location={self.output_dir}/segment_%05d.ts "
            f"playlist-location={self.hls_playlist} "
            f"max-files={self.max_files} "
            f"target-duration={self.hls_segment_time}"
        )

        logger.debug(f"GStreamer pipeline: {pipeline_str}")

        # Create pipeline
        self.pipeline = Gst.parse_launch(pipeline_str)
        logger.debug("Pipeline object created")

        # Get appsrc element
        self.appsrc = self.pipeline.get_by_name("source")
        logger.debug(f"appsrc element: {self.appsrc}")

        if not self.appsrc:
            raise RuntimeError("Failed to get appsrc element from pipeline")

        # Set appsrc properties for non-blocking behavior
        logger.debug("Setting appsrc properties...")
        self.appsrc.set_property("is-live", True)
        self.appsrc.set_property("format", Gst.Format.TIME)
        self.appsrc.set_property("block", False)  # Non-blocking push
        logger.debug("appsrc properties set")

        # Start pipeline
        logger.debug("Starting pipeline...")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        logger.debug(f"set_state returned: {ret}")

        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to start GStreamer pipeline")

        # Note: Pipeline may be in PAUSED state until first frame is pushed
        # This is normal for appsrc-based pipelines
        # The pipeline will automatically transition to PLAYING when data flows
        logger.info("✅ GStreamer pipeline started (will transition to PLAYING when data flows)")

    def write(self, frame: np.ndarray) -> bool:
        """
        Write a frame to GStreamer pipeline (non-blocking).

        Args:
            frame: BGR24 numpy array (height, width, 3)

        Returns:
            True if frame was sent, False if dropped
        """
        try:
            # Convert numpy array to GStreamer buffer
            frame_bytes = frame.tobytes()
            buf = Gst.Buffer.new_wrapped(frame_bytes)

            # Set buffer timestamp
            timestamp = self.frames_sent * Gst.SECOND // int(self.fps)
            buf.pts = timestamp
            buf.dts = timestamp
            buf.duration = Gst.SECOND // int(self.fps)

            # Push buffer to appsrc (non-blocking due to block=false)
            # Use emit signal instead of method (Python bindings don't have push_buffer method)
            ret = self.appsrc.emit("push-buffer", buf)

            if ret == Gst.FlowReturn.OK:
                self.frames_sent += 1

                # Log progress every 100 frames
                if self.frames_sent % 100 == 0:
                    logger.debug(f"🎬 GStreamer: {self.frames_sent} frames sent, {self.frames_dropped} dropped")

                return True
            else:
                # Frame was dropped (queue full)
                self.frames_dropped += 1

                # Log warning every 10 drops
                if self.frames_dropped % 10 == 0:
                    logger.warning(f"⚠️ GStreamer queue full, dropped {self.frames_dropped} frames total")

                return False

        except Exception as e:
            logger.error(f"Error pushing frame to GStreamer: {e}")
            return False

    def is_alive(self) -> bool:
        """
        Check if GStreamer pipeline is still running.

        Returns True if pipeline is in PAUSED or PLAYING state.
        PAUSED is normal for appsrc pipelines before first frame is pushed.
        """
        if self.pipeline is None:
            return False

        state = self.pipeline.get_state(0)  # Non-blocking check
        current_state = state[1]

        # Accept both PAUSED and PLAYING states
        # Pipeline will auto-transition from PAUSED to PLAYING when data flows
        is_alive = current_state in (Gst.State.PAUSED, Gst.State.PLAYING)

        return is_alive

    def release(self):
        """Stop GStreamer pipeline and cleanup"""
        try:
            if self.pipeline:
                # Send EOS to appsrc (use emit signal)
                self.appsrc.emit("end-of-stream")

                # Wait for EOS to propagate (max 5 seconds)
                bus = self.pipeline.get_bus()
                bus.timed_pop_filtered(5 * Gst.SECOND, Gst.MessageType.EOS | Gst.MessageType.ERROR)

                # Stop pipeline
                self.pipeline.set_state(Gst.State.NULL)

                logger.info(f"✅ GStreamer pipeline stopped")
                logger.info(f"   Total frames: {self.frames_sent} sent, {self.frames_dropped} dropped")

        except Exception as e:
            logger.error(f"Error stopping GStreamer pipeline: {e}")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()

