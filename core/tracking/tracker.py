"""
ByteTrack Tracker Wrapper
Multi-object tracking using ByteTrack algorithm
"""

import torch
import numpy as np
from loguru import logger

from yolox.tracker.byte_tracker import BYTETracker


class ByteTrackWrapper:
    """
    Wrapper for ByteTrack tracker
    Handles multi-object tracking with IoU + Kalman Filter
    """

    def __init__(self, track_thresh=0.5, track_buffer=30,
                 match_thresh=0.8, frame_rate=30, mot20=False):
        """
        Args:
            track_thresh: Tracking confidence threshold
            track_buffer: Frames to keep lost tracks
            match_thresh: Matching threshold for association
            frame_rate: Video frame rate
            mot20: Use MOT20 settings
        """
        logger.info("Initializing ByteTrack Tracker...")
        logger.info(f"  Track threshold: {track_thresh}")
        logger.info(f"  Track buffer: {track_buffer}")
        logger.info(f"  Match threshold: {match_thresh}")
        logger.info(f"  Frame rate: {frame_rate}")

        # Create args object for ByteTracker
        class Args:
            def __init__(self):
                self.track_thresh = track_thresh
                self.track_buffer = track_buffer
                self.match_thresh = match_thresh
                self.mot20 = mot20

        self.args = Args()
        self.tracker = BYTETracker(self.args, frame_rate=frame_rate)
        self.frame_count = 0

        logger.info("✅ Tracker initialized successfully")

    def update(self, detections, frame_shape):
        """
        Update tracker with new detections

        Args:
            detections: (N, 7) - [x1, y1, x2, y2, conf, cls, -1]
            frame_shape: (height, width)

        Returns:
            tracks: (M, 6) - [x1, y1, x2, y2, track_id, conf]
        """
        self.frame_count += 1

        if len(detections) == 0:
            return np.empty((0, 6))

        # Convert to torch tensor for ByteTracker
        detections_tensor = torch.from_numpy(detections).float()

        # Update tracker
        online_targets = self.tracker.update(
            detections_tensor,
            frame_shape,
            frame_shape
        )

        # Convert to output format
        tracks = []
        for t in online_targets:
            tlbr = t.tlbr
            tid = t.track_id
            conf = t.score
            tracks.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3], tid, conf])

        return np.array(tracks) if len(tracks) > 0 else np.empty((0, 6))

    def reset(self):
        """Reset tracker state"""
        self.tracker = BYTETracker(self.args, frame_rate=30)
        self.frame_count = 0
        logger.info("Tracker reset")

    def get_info(self):
        """Get tracker information"""
        return {
            'track_thresh': self.args.track_thresh,
            'track_buffer': self.args.track_buffer,
            'match_thresh': self.args.match_thresh,
            'mot20': self.args.mot20,
            'frame_count': self.frame_count,
        }
