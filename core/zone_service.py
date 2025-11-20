#!/usr/bin/env python3
"""
Zone Monitoring Service - Runs zone monitoring on separate threads (Thread Pool)
Decouples zone processing from main detection/tracking pipeline
Uses ThreadPoolExecutor for parallel zone processing
Integrated with Kafka for realtime alert streaming
"""

import threading
import queue
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import Kafka manager
try:
    from utils.kafka_manager import KafkaAlertProducer
    KAFKA_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ Kafka manager not available - alerts will not be sent to Kafka")
    logger.warning(f"   Error: {e}")
    logger.warning(f"   sys.path: {sys.path[:3]}")  # Show first 3 paths
    KAFKA_AVAILABLE = False
except Exception as e:
    logger.error(f"❌ Unexpected error importing Kafka manager: {e}")
    logger.error(f"   Error type: {type(e).__name__}")
    KAFKA_AVAILABLE = False


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

    def __init__(self, zone_monitor, max_queue_size: int = 100, num_workers: Optional[int] = None,
                 kafka_config: Optional[Dict] = None, alert_threshold: float = 0.0):
        """
        Args:
            zone_monitor: ZoneMonitor instance
            max_queue_size: Maximum queue size before dropping frames
            num_workers: Number of worker threads (default: CPU count)
                        Set to 1 for single-threaded mode (backward compatible)
            kafka_config: Kafka configuration dict with keys:
                         - bootstrap_servers: Kafka broker address
                         - topic: Topic name for alerts
                         - enable: Enable/disable Kafka
            alert_threshold: Time threshold (seconds) before triggering alert (default: 0 = immediate)
        """
        self.zone_monitor = zone_monitor
        self.alert_threshold = alert_threshold

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

        # Violation tracking (for threshold-based alerts)
        # {zone_id: {'start_time': float, 'alerted': bool, 'missing_persons': list}}
        self.violation_tracker = {}

        # Kafka Producer
        self.kafka_producer = None
        logger.info(f"🔍 Kafka init check: KAFKA_AVAILABLE={KAFKA_AVAILABLE}, kafka_config={kafka_config is not None}")
        if kafka_config:
            logger.info(f"   kafka_config.enable={kafka_config.get('enable', False)}")

        if KAFKA_AVAILABLE and kafka_config and kafka_config.get('enable', False):
            try:
                self.kafka_producer = KafkaAlertProducer(
                    bootstrap_servers=kafka_config.get('bootstrap_servers', 'localhost:9092'),
                    topic=kafka_config.get('topic', 'person_reid_alerts'),
                    enable=True
                )
                logger.info(f"✅ Kafka Producer enabled for zone alerts (threshold: {alert_threshold}s)")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Kafka Producer: {e}")
                self.kafka_producer = None
        else:
            logger.info(f"⚠️ Kafka Producer disabled for zone alerts (KAFKA_AVAILABLE={KAFKA_AVAILABLE}, kafka_config={kafka_config is not None})")
        
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

        # Close Kafka producer
        if self.kafka_producer:
            self.kafka_producer.close()

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
        Process zone monitoring task with Kafka alerts

        Args:
            task: ZoneTask with frame data

        Returns:
            ZoneResult with zone status and violations
        """
        zone_results = {}

        # Update presence for each track
        for track in task.tracks:
            _, _, _, _, track_id, _ = track  # Unpack but only use track_id
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

        # Update all zones and check for violations
        new_violations = []
        for zone_id in self.zone_monitor.zones.keys():
            self.zone_monitor._update_zone_status(zone_id, task.frame_time)

            # Check violation with threshold
            violation = self._check_zone_violation(zone_id, task.frame_time, task.camera_idx, task.frame_id)
            if violation:
                new_violations.append(violation)

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

    def _check_zone_violation(self, zone_id: int, frame_time: float, camera_idx: int, frame_id: int) -> Optional[Dict]:
        """
        Check if zone violation and publish to Kafka (always, regardless of threshold)
        Threshold is handled by Consumer side for filtering/alerting

        Args:
            zone_id: Zone ID to check
            frame_time: Current frame time
            camera_idx: Camera index
            frame_id: Frame ID

        Returns:
            Violation dict if violation detected, None otherwise
        """
        zone_state = self.zone_monitor.zone_status[zone_id]
        zone_data = self.zone_monitor.zones[zone_id]

        # If zone is complete, clear violation tracker
        if zone_state['is_complete']:
            if zone_id in self.violation_tracker:
                del self.violation_tracker[zone_id]
            return None

        # Zone is incomplete - violation detected
        missing_persons = zone_state['missing_persons']

        # Initialize violation tracker for this zone
        if zone_id not in self.violation_tracker:
            self.violation_tracker[zone_id] = {
                'start_time': frame_time,
                'missing_persons': missing_persons.copy()
            }

        tracker = self.violation_tracker[zone_id]
        violation_duration = frame_time - tracker['start_time']

        # Get missing person names
        missing_names = []
        for pid in missing_persons:
            if pid in self.zone_monitor.person_locations:
                missing_names.append(self.zone_monitor.person_locations[pid]['name'])
            else:
                missing_names.append(f"Person {pid}")

        # ALWAYS publish to Kafka EVERY FRAME (don't wait for threshold)
        # Consumer will handle threshold-based filtering and deduplication
        if self.kafka_producer:
            for pid, name in zip(missing_persons, missing_names):
                self.kafka_producer.send_alert(
                    user_id=str(pid),
                    user_name=name,
                    camera_id=camera_idx,
                    zone_id=zone_id,
                    zone_name=zone_state['name'],
                    iop=self.zone_monitor.iop_threshold,  # Zone IoP threshold (e.g., 0.6 = 60%)
                    threshold=self.alert_threshold,  # Alert time threshold for Consumer reference
                    status='violation_incomplete',
                    frame_id=frame_id,
                    additional_data={
                        'violation_duration': round(violation_duration, 2),
                        'missing_count': len(missing_persons),
                        'required_count': len(zone_state['required_persons']),
                        'violation_start_time': tracker['start_time']
                    }
                )

            # Return violation for legacy compatibility
            return {
                'type': 'zone_incomplete',
                'zone_id': zone_id,
                'zone_name': zone_state['name'],
                'missing_persons': missing_persons,
                'missing_names': missing_names,
                'time': frame_time,
                'duration': violation_duration,
                'camera_idx': camera_idx
            }

        return None
        
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

