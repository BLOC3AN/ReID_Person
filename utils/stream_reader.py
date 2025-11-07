#!/usr/bin/env python3
"""
Stream Reader Module
Handles various video stream sources (UDP, RTSP, files) with robust error handling
"""

import cv2
import subprocess
import numpy as np
import time
import select
import sys
from loguru import logger
from typing import Optional, Tuple


class StreamReader:
    """
    Unified interface for reading video streams and files.
    Handles problematic streams (missing SPS/PPS headers) using ffmpeg subprocess.
    """
    
    def __init__(self, source: str, use_ffmpeg_for_udp: bool = True):
        """
        Initialize stream reader.
        
        Args:
            source: Video source (file path, UDP URL, RTSP URL, etc.)
            use_ffmpeg_for_udp: Use ffmpeg subprocess for UDP streams (more reliable)
        """
        self.source = source
        self.use_ffmpeg_for_udp = use_ffmpeg_for_udp
        self.is_stream = source.startswith(('udp://', 'rtsp://', 'rtmp://', 'http://', 'https://'))
        self.is_udp = source.startswith('udp://')
        
        # Stream properties
        self.width = None
        self.height = None
        self.fps = None
        
        # OpenCV or ffmpeg backend
        self.cap = None
        self.ffmpeg_process = None
        self.using_ffmpeg = False
        
        # Initialize the stream
        self._initialize()
    
    def _initialize(self):
        """Initialize the video source."""
        if self.is_udp and self.use_ffmpeg_for_udp:
            self._initialize_ffmpeg_udp()
        else:
            self._initialize_opencv()
    
    def _initialize_opencv(self):
        """Initialize using OpenCV VideoCapture."""
        logger.info(f"Opening {'stream' if self.is_stream else 'video file'}: {self.source}")
        
        if self.is_stream:
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            time.sleep(2)  # Wait for stream to stabilize
        else:
            self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open source: {self.source}")
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        if self.fps == 0 or self.fps is None:
            self.fps = 25.0  # Default for streams
        
        logger.info(f"✓ Opened via OpenCV: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        self.using_ffmpeg = False
    
    def _initialize_ffmpeg_udp(self):
        """Initialize UDP stream using ffmpeg subprocess (more reliable)."""
        logger.info(f"Opening UDP stream via ffmpeg: {self.source}")
        logger.info("This method is more reliable for streams with missing SPS/PPS headers")
        
        # For UDP streams, skip probing (ffprobe will also try to bind the port)
        # Use default values instead
        logger.info("Skipping probe for UDP stream (would cause port binding conflict)")
        logger.info("Using default stream properties: 1920x1080 @ 25 FPS")
        self.width = 1920
        self.height = 1080
        self.fps = 25.0
        
        logger.info(f"Stream properties: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        
        # Add UDP-specific options to the source URL
        source_url = self.source
        if 'udp://' in self.source and '?' not in self.source:
            # Add options to prevent binding and handle packet loss
            source_url = f"{self.source}?overrun_nonfatal=1&fifo_size=50000000"

        # Build ffmpeg command to decode and output raw BGR24 frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-timeout', '5000000',  # 5 second timeout for network operations (in microseconds)
            '-fflags', '+genpts+discardcorrupt',
            '-flags', 'low_delay',
            '-probesize', '32M',  # Increased probe size
            '-analyzeduration', '5M',  # Some analysis to get codec info
            '-i', source_url,
            '-vsync', '0',  # Pass through timestamps, don't wait for missing frames
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-an',  # No audio
            '-sn',  # No subtitles
            'pipe:1'
        ]
        
        logger.info(f"Starting ffmpeg subprocess...")
        logger.debug(f"Command: {' '.join(ffmpeg_cmd)}")

        # Start ffmpeg process
        # IMPORTANT: Redirect stderr to DEVNULL to prevent pipe buffer overflow
        # ffmpeg writes A LOT of logs to stderr (decode errors, warnings, etc.)
        # If stderr pipe fills up (~65KB), ffmpeg will BLOCK and stop writing to stdout
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Prevent stderr buffer overflow!
            bufsize=10**8
        )
        
        # Wait for ffmpeg to start and begin decoding
        logger.info("Waiting for ffmpeg to start decoding (5 seconds)...")
        time.sleep(5)

        # Check if process is still running
        if self.ffmpeg_process.poll() is not None:
            raise RuntimeError(f"ffmpeg process terminated unexpectedly (exit code: {self.ffmpeg_process.returncode})")
        
        logger.info(f"✓ Stream opened via ffmpeg subprocess: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        self.using_ffmpeg = True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the stream.
        
        Returns:
            Tuple of (success, frame)
            - success: True if frame was read successfully
            - frame: numpy array (H, W, 3) in BGR format, or None if failed
        """
        if self.using_ffmpeg:
            return self._read_ffmpeg()
        else:
            return self._read_opencv()
    
    def _read_opencv(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame using OpenCV."""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        return ret, frame
    
    def _read_ffmpeg(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read frame from ffmpeg subprocess."""
        if self.ffmpeg_process is None:
            return False, None
        
        # Check if process is still running
        if self.ffmpeg_process.poll() is not None:
            logger.error("ffmpeg process has terminated")
            return False, None
        
        # Calculate frame size
        frame_size = self.width * self.height * 3  # BGR24 = 3 bytes per pixel

        try:
            # Check if data is available with timeout (3 seconds)
            # Use select on Unix-like systems
            if sys.platform != 'win32':
                ready, _, _ = select.select([self.ffmpeg_process.stdout], [], [], 3.0)
                if not ready:
                    logger.warning("No frame available from ffmpeg (3s timeout) - stream may be lagging, skipping")
                    return False, None

            # Read raw frame data
            # ffmpeg has -timeout flag, so it will fail if stream is stuck
            raw_frame = self.ffmpeg_process.stdout.read(frame_size)

            if len(raw_frame) == 0:
                logger.warning("ffmpeg returned 0 bytes - stream ended or error")
                return False, None

            if len(raw_frame) != frame_size:
                logger.warning(f"Incomplete frame: expected {frame_size} bytes, got {len(raw_frame)} - skipping")
                # Try to flush remaining bytes to resync
                return False, None

            # Convert raw bytes to numpy array
            # Use np.frombuffer which creates a readonly array, then copy to make it writable
            frame = np.frombuffer(raw_frame, dtype=np.uint8).copy()
            frame = frame.reshape((self.height, self.width, 3))

            return True, frame

        except Exception as e:
            logger.error(f"Error reading frame from ffmpeg: {e}")
            return False, None
    
    def get_properties(self) -> dict:
        """Get stream properties."""
        return {
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'source': self.source,
            'is_stream': self.is_stream,
            'using_ffmpeg': self.using_ffmpeg
        }
    
    def release(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        if self.ffmpeg_process is not None:
            logger.info("Terminating ffmpeg subprocess...")
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("ffmpeg did not terminate gracefully, killing...")
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()

