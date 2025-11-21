#!/usr/bin/env python3
"""
Live Stream Writer
Write processed frames to HLS livestream via FFmpeg pipe for real-time viewing
"""

import subprocess
import numpy as np
import cv2
from loguru import logger
from typing import Optional
import os
from pathlib import Path


class LiveStreamWriter:
    """
    Write processed frames to HLS livestream via FFmpeg pipe
    
    This class enables real-time streaming of AI-processed video frames
    (with bounding boxes, labels, tracking) to web browsers via HLS protocol.
    
    Architecture:
        Python frames → FFmpeg stdin → HLS segments → Flask server → Browser
    """
    
    def __init__(self, output_dir: str, width: int, height: int, fps: float,
                 hls_segment_time: int = 6, hls_list_size: int = 10):
        """
        Initialize livestream writer

        Args:
            output_dir: Directory to save HLS files (stream.m3u8 and segment_*.ts)
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Frames per second
            hls_segment_time: Duration of each HLS segment in seconds (default: 6)
            hls_list_size: Number of segments to keep in playlist (default: 10)
        """
        self.output_dir = Path(output_dir)
        self.width = width
        self.height = height
        self.fps = fps
        self.hls_segment_time = hls_segment_time
        self.hls_list_size = hls_list_size
        self.ffmpeg_process = None
        self.frames_written = 0

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # HLS output path
        self.hls_playlist = self.output_dir / "stream.m3u8"

        # Start FFmpeg process
        self._start_ffmpeg()
    
    def _start_ffmpeg(self):
        """Start FFmpeg process with pipe input for HLS output"""
        
        # FFmpeg command for smooth HLS streaming:
        # - Input: raw BGR frames from stdin (OpenCV format)
        # - Output: HLS stream with configurable segments
        # - Preset: ultrafast for low CPU usage
        buffer_time = self.hls_segment_time * self.hls_list_size
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
            '-s', f'{self.width}x{self.height}',
            '-r', str(self.fps),
            '-i', '-',  # Read from stdin
            '-c:v', 'libx264',
            '-preset', 'ultrafast',  # Fast encoding (low CPU)
            '-tune', 'zerolatency',  # Low latency
            '-g', str(int(self.fps * self.hls_segment_time)),  # Keyframe interval
            '-sc_threshold', '0',  # Disable scene change detection
            '-f', 'hls',
            '-hls_time', str(self.hls_segment_time),  # Segment duration
            '-hls_list_size', str(self.hls_list_size),  # Playlist size
            '-hls_flags', 'delete_segments+append_list',  # Auto-delete old segments
            '-hls_segment_filename', str(self.output_dir / 'segment_%03d.ts'),
            str(self.hls_playlist)
        ]

        logger.info(f"🎬 Starting FFmpeg livestream writer")
        logger.info(f"   Output: {self.hls_playlist}")
        logger.info(f"   Resolution: {self.width}x{self.height} @ {self.fps:.1f} FPS")
        logger.info(f"   Segment: {self.hls_segment_time}s, Playlist: {self.hls_list_size} segments")
        logger.info(f"   Buffer: {buffer_time}s total for smooth playback")
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8  # Large buffer (100MB) for smooth streaming
            )
            logger.info("✅ FFmpeg livestream writer started successfully")
        except Exception as e:
            logger.error(f"❌ Failed to start FFmpeg: {e}")
            raise
    
    def write(self, frame: np.ndarray) -> bool:
        """
        Write a processed frame to livestream
        
        Args:
            frame: BGR frame (numpy array, shape: [height, width, 3])
            
        Returns:
            True if successful, False otherwise
        """
        if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
            logger.error("FFmpeg process not running")
            return False
        
        try:
            # Write raw frame bytes to FFmpeg stdin
            self.ffmpeg_process.stdin.write(frame.tobytes())
            self.frames_written += 1
            
            # Log progress every 100 frames
            if self.frames_written % 100 == 0:
                logger.debug(f"📡 Livestream: {self.frames_written} frames written")
            
            return True
        except BrokenPipeError:
            logger.error("FFmpeg pipe broken - process may have crashed")
            return False
        except Exception as e:
            logger.error(f"Error writing frame to FFmpeg: {e}")
            return False
    
    def release(self):
        """Close FFmpeg process and cleanup"""
        if self.ffmpeg_process:
            logger.info(f"🛑 Stopping FFmpeg livestream writer ({self.frames_written} frames written)")
            try:
                # Close stdin to signal end of stream
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Wait for FFmpeg to finish processing
                self.ffmpeg_process.wait(timeout=5)
                logger.info("✅ FFmpeg livestream writer stopped cleanly")
            except subprocess.TimeoutExpired:
                logger.warning("FFmpeg did not stop in time, killing process")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
            except Exception as e:
                logger.error(f"Error stopping FFmpeg: {e}")
    
    def is_alive(self) -> bool:
        """Check if FFmpeg process is still running"""
        return self.ffmpeg_process is not None and self.ffmpeg_process.poll() is None
    
    def __del__(self):
        """Cleanup on object destruction"""
        self.release()

