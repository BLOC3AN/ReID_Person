#!/usr/bin/env python3
"""
Zone Monitoring Service - Runs zone monitoring on separate threads (Thread Pool)
Decouples zone processing from main detection/tracking pipeline
Uses ThreadPoolExecutor for parallel zone processing
"""

import threading
import queue
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import os


@dataclass
class ZoneTask:
    """Task data for zone monitoring thread"""
    frame_id: int
    frame_time: float
    tracks: List[tuple]  # [(x1,y1,x2,y2,track_id,conf), ...]
    reid_results: Dict[int, Dict]  # {track_id: {global_id, name, ...}}
    zone_ids: Dict[int, Optional[int]]  # {track_id: zone_id} - pre-computed
    camera_idx: int = 0


@dataclass
class ZoneResult:
    """Result data from zone monitoring thread"""
    frame_id: int
    frame_time: float
    zone_status: Dict[int, Dict]  # Zone status for each zone
    zone_results: Dict[int, Dict]  # {track_id: {zone_id, zone_name}}
    violations: List[Dict]  # New violations in this frame
    process_time_ms: float


class ZoneMonitoringService:
    """
    Service for zone monitoring on thread pool

    Receives pre-computed zone_ids from main thread
    Processes zone status updates and violation detection using multiple threads
    Returns results via queue for main thread to use
    """

    def __init__(self, zone_monitor, max_queue_size: int = 100, num_workers: Optional[int] = None):
        """
        Args:
            zone_monitor: ZoneMonitor instance
            max_queue_size: Maximum queue size before dropping frames
            num_workers: Number of worker threads (default: CPU count)
                        Set to 1 for single-threaded mode (backward compatible)
        """
        self.zone_monitor = zone_monitor

        # Determine number of workers
        if num_workers is None:
            # Default: use CPU count, but cap at 4 for zone processing
            num_workers = min(os.cpu_count() or 2, 4)

        self.num_workers = num_workers

        # Queues for communication
        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)

        # Thread pool
        self.executor = None
        self.running = False
        self.threads = []

        # Metrics
        self.processed_frames = 0
        self.dropped_frames = 0
        self.avg_process_time = 0.0
        
    def start(self):
        """Start zone monitoring thread pool"""
        if self.running:
            logger.warning("Zone service already running")
            return

        self.running = True

        # Create thread pool executor
        self.executor = ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="ZoneWorker"
        )

        # Start dispatcher thread (reads from input queue and submits to pool)
        dispatcher_thread = threading.Thread(
            target=self._dispatcher_loop,
            daemon=True,
            name="ZoneDispatcher"
        )
        dispatcher_thread.start()
        self.threads.append(dispatcher_thread)

        logger.info(f"✅ Zone monitoring service started with {self.num_workers} worker threads")
        
    def stop(self):
        """Stop zone monitoring thread pool"""
        if not self.running:
            return

        self.running = False

        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)

        # Wait for dispatcher thread
        for thread in self.threads:
            thread.join(timeout=2.0)

        logger.info("🛑 Zone monitoring service stopped")
        
    def submit_task(self, task: ZoneTask) -> bool:
        """
        Submit zone monitoring task (non-blocking)
        
        Args:
            task: ZoneTask with frame data
            
        Returns:
            True if submitted, False if queue full
        """
        try:
            self.input_queue.put(task, block=False)
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False
            
    def get_result(self, timeout: float = 0.001) -> Optional[ZoneResult]:
        """
        Get zone monitoring result (non-blocking)
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            ZoneResult if available, None otherwise
        """
        try:
            return self.result_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
            
    def _dispatcher_loop(self):
        """Dispatcher loop - reads tasks and submits to thread pool"""
        logger.info("Zone dispatcher started")

        while self.running:
            try:
                # Get task (blocking with timeout to check running flag)
                task = self.input_queue.get(timeout=0.1)

                # Submit task to thread pool
                future = self.executor.submit(self._process_zone_task_wrapper, task)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in zone dispatcher: {e}", exc_info=True)

        logger.info("Zone dispatcher stopped")

    def _process_zone_task_wrapper(self, task: ZoneTask):
        """Wrapper for zone task processing (runs in thread pool)"""
        try:
            start_time = time.time()
            result = self._process_zone_task(task)
            process_time = time.time() - start_time

            # Update metrics
            self.processed_frames += 1
            self.avg_process_time = self.avg_process_time * 0.9 + process_time * 0.1

            # Send result
            result.process_time_ms = process_time * 1000

            try:
                self.result_queue.put(result, block=False)
            except queue.Full:
                pass  # Drop result if queue full

        except Exception as e:
            logger.error(f"Error processing zone task: {e}", exc_info=True)
        
    def _process_zone_task(self, task: ZoneTask) -> ZoneResult:
        """
        Process zone monitoring task
        
        Args:
            task: ZoneTask with frame data
            
        Returns:
            ZoneResult with zone status and violations
        """
        zone_results = {}
        violations_before = len(self.zone_monitor.zone_violations)
        
        # Update presence for each track
        for track in task.tracks:
            x1, y1, x2, y2, track_id, conf = track
            track_id = int(track_id)
            
            reid_info = task.reid_results.get(track_id)
            if not reid_info or reid_info['global_id'] <= 0:
                continue
            
            # Use pre-computed zone_id from main thread
            zone_id = task.zone_ids.get(track_id)
            
            # Update presence
            self.zone_monitor.update_presence(
                reid_info['global_id'],
                zone_id,
                task.frame_time,
                reid_info['person_name']
            )
            
            zone_results[track_id] = {
                'zone_id': zone_id,
                'zone_name': self.zone_monitor.zones[zone_id]['name'] if zone_id else None
            }
        
        # Update all zones
        for zone_id in self.zone_monitor.zones.keys():
            self.zone_monitor._update_zone_status(zone_id, task.frame_time)
        
        # Get new violations
        violations_after = len(self.zone_monitor.zone_violations)
        new_violations = []
        if violations_after > violations_before:
            new_violations = self.zone_monitor.zone_violations[violations_before:violations_after]
        
        # Copy zone status (avoid race conditions)
        zone_status_copy = {}
        for zone_id, zone_state in self.zone_monitor.zone_status.items():
            zone_status_copy[zone_id] = {
                'name': zone_state['name'],
                'is_complete': zone_state['is_complete'],
                'present_persons': zone_state['present_persons'].copy(),
                'missing_persons': zone_state['missing_persons'].copy()
            }
        
        return ZoneResult(
            frame_id=task.frame_id,
            frame_time=task.frame_time,
            zone_status=zone_status_copy,
            zone_results=zone_results,
            violations=new_violations,
            process_time_ms=0.0
        )
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            'num_workers': self.num_workers,
            'processed_frames': self.processed_frames,
            'dropped_frames': self.dropped_frames,
            'avg_process_time_ms': self.avg_process_time * 1000,
            'input_queue_size': self.input_queue.qsize(),
            'result_queue_size': self.result_queue.qsize(),
            'running': self.running
        }

