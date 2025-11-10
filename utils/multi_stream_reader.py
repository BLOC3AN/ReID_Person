#!/usr/bin/env python3
"""
Multi-Stream Reader Module
Handles multiple video streams simultaneously with frame synchronization
"""

import cv2
import numpy as np
import threading
import queue
import time
from typing import List, Optional, Tuple, Dict
from loguru import logger
from .stream_reader import StreamReader


class MultiStreamReader:
    """
    Read from multiple video streams simultaneously and provide synchronized frames.
    Supports parallel reading with frame buffering and synchronization.
    """

    def __init__(self, sources: List[str], use_ffmpeg_for_udp: bool = True, 
                 buffer_size: int = 30, sync_tolerance: float = 0.1):
        """
        Initialize multi-stream reader.

        Args:
            sources: List of video sources (file paths, UDP URLs, RTSP URLs, etc.)
            use_ffmpeg_for_udp: Use ffmpeg subprocess for UDP streams
            buffer_size: Maximum frames to buffer per stream
            sync_tolerance: Time tolerance for frame synchronization (seconds)
        """
        if not sources:
            raise ValueError("At least one source must be provided")
        
        self.sources = sources
        self.num_streams = len(sources)
        self.use_ffmpeg_for_udp = use_ffmpeg_for_udp
        self.buffer_size = buffer_size
        self.sync_tolerance = sync_tolerance
        
        # Stream readers
        self.readers: List[StreamReader] = []
        self.reader_threads: List[threading.Thread] = []
        self.frame_queues: List[queue.Queue] = []
        
        # Control flags
        self.running = False
        self.stop_flag = threading.Event()
        
        # Stream properties (use first stream as reference)
        self.width = None
        self.height = None
        self.fps = None
        self.is_stream = False
        
        # Initialize streams
        self._initialize()
    
    def _initialize(self):
        """Initialize all stream readers and start reading threads."""
        logger.info(f"Initializing {self.num_streams} streams...")
        
        # Create readers for each source
        for i, source in enumerate(self.sources):
            try:
                logger.info(f"Stream {i+1}/{self.num_streams}: {source}")
                reader = StreamReader(source, use_ffmpeg_for_udp=self.use_ffmpeg_for_udp)
                self.readers.append(reader)
                
                # Create frame queue for this stream
                self.frame_queues.append(queue.Queue(maxsize=self.buffer_size))
                
                # Get properties from first stream
                if i == 0:
                    props = reader.get_properties()
                    self.width = props['width']
                    self.height = props['height']
                    self.fps = props['fps']
                    self.is_stream = props['is_stream']
                    logger.info(f"Reference stream properties: {self.width}x{self.height} @ {self.fps:.1f} FPS")
                
            except Exception as e:
                logger.error(f"Failed to initialize stream {i+1}: {e}")
                # Cleanup already initialized readers
                for reader in self.readers:
                    reader.release()
                raise RuntimeError(f"Failed to initialize stream {i+1}: {e}")
        
        # Start reading threads
        self.running = True
        for i, reader in enumerate(self.readers):
            thread = threading.Thread(
                target=self._read_stream_worker,
                args=(i, reader, self.frame_queues[i]),
                daemon=True,
                name=f"StreamReader-{i}"
            )
            thread.start()
            self.reader_threads.append(thread)
        
        logger.info(f"✓ All {self.num_streams} streams initialized and reading started")
    
    def _read_stream_worker(self, stream_id: int, reader: StreamReader, frame_queue: queue.Queue):
        """
        Worker thread to continuously read frames from a stream.
        
        Args:
            stream_id: Stream identifier
            reader: StreamReader instance
            frame_queue: Queue to put frames into
        """
        logger.debug(f"Stream {stream_id} worker started")
        consecutive_failures = 0
        max_failures = 10
        
        while self.running and not self.stop_flag.is_set():
            try:
                ret, frame = reader.read()
                
                if ret and frame is not None:
                    consecutive_failures = 0
                    timestamp = time.time()
                    
                    # Try to put frame in queue (non-blocking)
                    try:
                        frame_queue.put((timestamp, frame), block=False)
                    except queue.Full:
                        # Queue is full, drop oldest frame
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put((timestamp, frame), block=False)
                        except:
                            pass
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"Stream {stream_id} failed {consecutive_failures} times, stopping")
                        break
                    time.sleep(0.01)  # Brief pause before retry
                    
            except Exception as e:
                logger.error(f"Stream {stream_id} read error: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    break
                time.sleep(0.1)
        
        logger.debug(f"Stream {stream_id} worker stopped")
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read synchronized frames from all streams and combine them.
        
        Returns:
            Tuple of (success, combined_frame)
            - success: True if frames were read successfully from all streams
            - combined_frame: Horizontally concatenated frames from all streams
        """
        if not self.running:
            return False, None
        
        frames = []
        timestamps = []
        
        # Try to get a frame from each stream
        for i, frame_queue in enumerate(self.frame_queues):
            try:
                # Wait up to 1 second for a frame
                timestamp, frame = frame_queue.get(timeout=1.0)
                frames.append(frame)
                timestamps.append(timestamp)
            except queue.Empty:
                logger.warning(f"Stream {i} queue empty (timeout)")
                return False, None
        
        # Check if we got frames from all streams
        if len(frames) != self.num_streams:
            return False, None
        
        # Resize frames to same height if needed
        target_height = self.height
        resized_frames = []
        for frame in frames:
            if frame.shape[0] != target_height:
                aspect_ratio = frame.shape[1] / frame.shape[0]
                target_width = int(target_height * aspect_ratio)
                resized = cv2.resize(frame, (target_width, target_height))
                resized_frames.append(resized)
            else:
                resized_frames.append(frame)
        
        # Concatenate frames horizontally
        combined_frame = np.hstack(resized_frames)
        
        return True, combined_frame
    
    def get_properties(self) -> dict:
        """Get combined stream properties."""
        # Calculate combined width (sum of all stream widths)
        combined_width = self.width * self.num_streams
        
        return {
            'width': combined_width,
            'height': self.height,
            'fps': self.fps,
            'source': f"Multi-stream ({self.num_streams} cameras)",
            'sources': self.sources,
            'is_stream': self.is_stream,
            'using_ffmpeg': any(r.using_ffmpeg for r in self.readers),
            'num_streams': self.num_streams
        }
    
    def release(self):
        """Release all resources."""
        logger.info("Releasing multi-stream reader...")
        
        # Stop reading threads
        self.running = False
        self.stop_flag.set()
        
        # Wait for threads to finish (with timeout)
        for i, thread in enumerate(self.reader_threads):
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(f"Stream {i} thread did not stop gracefully")
        
        # Release all readers
        for reader in self.readers:
            reader.release()
        
        # Clear queues
        for frame_queue in self.frame_queues:
            while not frame_queue.empty():
                try:
                    frame_queue.get_nowait()
                except:
                    break
        
        logger.info("✓ Multi-stream reader released")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
    
    def __del__(self):
        """Destructor."""
        self.release()


def parse_stream_urls(url_string: str) -> List[str]:
    """
    Parse multiple stream URLs from a string.
    
    Supports formats:
    - Single URL: "udp://127.0.0.1:1905"
    - Multiple URLs (comma-separated): "udp://127.0.0.1:1905, udp://127.0.0.1:1906"
    - Multiple URLs (newline-separated): "udp://127.0.0.1:1905\nudp://127.0.0.1:1906"
    
    Args:
        url_string: String containing one or more URLs
        
    Returns:
        List of cleaned URLs
    """
    # Split by comma or newline
    urls = []
    for separator in [',', '\n']:
        if separator in url_string:
            urls = [url.strip() for url in url_string.split(separator)]
            break
    
    # If no separator found, treat as single URL
    if not urls:
        urls = [url_string.strip()]
    
    # Filter out empty strings
    urls = [url for url in urls if url]
    
    return urls

