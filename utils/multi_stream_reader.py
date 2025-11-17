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

        # Warmup: Wait for all streams to have at least one valid frame in queue
        # UDP H.264 streams often have decode errors at start (missing PPS/SPS)
        # We just need to wait for the first keyframe (I-frame) which typically comes every 1-2 seconds
        logger.info("⏳ Warming up streams, waiting for first valid frames...")
        warmup_timeout = 3.0  # 3 seconds is enough for 1-2 keyframes at typical GOP size
        warmup_start = time.time()

        while time.time() - warmup_start < warmup_timeout:
            all_ready = all(not q.empty() for q in self.frame_queues)
            if all_ready:
                logger.info(f"✓ All streams ready after {time.time() - warmup_start:.1f}s")
                break
            time.sleep(0.1)
        else:
            # Check which streams are not ready
            not_ready = [i for i, q in enumerate(self.frame_queues) if q.empty()]
            if not_ready:
                logger.warning(f"⚠️ Streams {not_ready} not ready after {warmup_timeout}s warmup (may have H.264 decode errors)")
                logger.info("   This is normal for UDP streams - will retry on first read()")
            else:
                logger.info("✓ All streams ready")
    
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

        # Try to get a frame from each stream with retry logic
        # For UDP streams with H.264 decode errors at start, we need more retries with shorter timeout
        max_retries = 5  # Increased from 3 to 5
        timeout_per_try = 2.0  # Reduced from 5.0 to 2.0 seconds (faster retry)

        for i, frame_queue in enumerate(self.frame_queues):
            frame_received = False

            for retry in range(max_retries):
                try:
                    # Wait up to 2 seconds for a frame
                    timestamp, frame = frame_queue.get(timeout=timeout_per_try)
                    frames.append(frame)
                    timestamps.append(timestamp)
                    frame_received = True
                    break
                except queue.Empty:
                    if retry < max_retries - 1:
                        logger.debug(f"Stream {i} queue empty (retry {retry + 1}/{max_retries})")
                    else:
                        logger.warning(f"Stream {i} queue empty after {max_retries} retries (total {max_retries * timeout_per_try}s)")

            if not frame_received:
                # Only return False if we've exhausted all retries
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
            'num_streams': self.num_streams,
            'single_camera_width': self.width  # Width of each individual camera
        }

    def get_camera_index(self, x_coord: int) -> int:
        """
        Get camera index from x-coordinate in combined frame.

        Args:
            x_coord: X-coordinate in combined frame

        Returns:
            Camera index (0-based)
        """
        camera_idx = min(x_coord // self.width, self.num_streams - 1)
        return camera_idx

    def bbox_to_camera_relative(self, bbox: list) -> Tuple[list, int]:
        """
        Convert bbox from combined frame coordinates to camera-relative coordinates.

        Args:
            bbox: [x, y, w, h] or [x1, y1, x2, y2] in combined frame

        Returns:
            Tuple of (relative_bbox, camera_idx)
            - relative_bbox: bbox with x-coordinates relative to camera frame
            - camera_idx: which camera this bbox belongs to
        """
        # Detect format
        if len(bbox) == 4:
            x1, y1, x2_or_w, y2_or_h = bbox

            # Check if it's xywh or xyxy
            if x2_or_w < x1 or y2_or_h < y1:
                # Likely xywh format (w/h are small)
                is_xywh = True
            elif x2_or_w > self.width * self.num_streams or y2_or_h > self.height:
                # x2/y2 are beyond single camera, so it's xyxy
                is_xywh = False
            else:
                # Ambiguous, assume xyxy if x2 > x1
                is_xywh = x2_or_w <= self.width

            if is_xywh:
                x, y, w, h = bbox
                x2 = x + w
            else:
                x, y, x2, y2_or_h = bbox
                w = x2 - x
                h = y2_or_h - y
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")

        # Determine camera index from x-coordinate
        camera_idx = self.get_camera_index(x)

        # Convert to camera-relative coordinates
        camera_x_offset = camera_idx * self.width
        relative_x = x - camera_x_offset
        relative_x2 = x2 - camera_x_offset

        # Clip to camera bounds
        relative_x = max(0, min(relative_x, self.width))
        relative_x2 = max(0, min(relative_x2, self.width))

        # Return in same format as input
        if is_xywh:
            relative_bbox = [relative_x, y, relative_x2 - relative_x, h]
        else:
            relative_bbox = [relative_x, y, relative_x2, y2_or_h]

        return relative_bbox, camera_idx
    
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

