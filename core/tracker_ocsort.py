"""
OC-SORT Tracker Wrapper
Multi-object tracking using OC-SORT (Observation-Centric SORT)
Pure motion-based tracker with Kalman Filter + IoU matching
"""

import numpy as np
from loguru import logger

from core.ocsort import OCSort


class OCSortWrapper:
    """
    Wrapper for OC-SORT tracker
    Handles multi-object tracking with observation-centric re-update
    Better for crowded scenes and non-linear motion
    """

    def __init__(self, det_thresh=0.5, max_age=30, min_hits=3,
                 iou_threshold=0.3, delta_t=3, asso_func="iou",
                 inertia=0.2, use_byte=False):
        """
        Args:
            det_thresh: Detection confidence threshold
            max_age: Maximum frames to keep lost tracks
            min_hits: Minimum hits to confirm track
            iou_threshold: IoU threshold for association
            delta_t: Time delta for observation-centric update
            asso_func: Association function (iou, giou, ciou, diou, ct_dist)
            inertia: Inertia factor for velocity
            use_byte: Use BYTE-style second matching
        """
        logger.info("Initializing OC-SORT Tracker...")
        logger.info(f"  Detection threshold: {det_thresh}")
        logger.info(f"  Max age: {max_age}")
        logger.info(f"  Min hits: {min_hits}")
        logger.info(f"  IoU threshold: {iou_threshold}")
        logger.info(f"  Delta t: {delta_t}")
        logger.info(f"  Association function: {asso_func}")
        logger.info(f"  Inertia: {inertia}")
        logger.info(f"  Use BYTE: {use_byte}")

        self.tracker = OCSort(
            det_thresh=det_thresh,
            max_age=max_age,
            min_hits=min_hits,
            iou_threshold=iou_threshold,
            delta_t=delta_t,
            asso_func=asso_func,
            inertia=inertia,
            use_byte=use_byte
        )
        self.frame_count = 0
        self.det_thresh = det_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.delta_t = delta_t
        self.asso_func = asso_func
        self.inertia = inertia
        self.use_byte = use_byte

        logger.info("✅ OC-SORT Tracker initialized successfully")

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

        # Convert from (N, 7) [x1, y1, x2, y2, conf, cls, -1]
        # to (N, 5) [x1, y1, x2, y2, conf] for OC-SORT
        if len(detections) == 0:
            dets = np.empty((0, 5))
        else:
            dets = detections[:, :5].copy()

        # OC-SORT expects:
        # - output_results: (N, 5) [x1, y1, x2, y2, score] - already in original image coords
        # - img_info: (height, width) - original image size
        # - img_size: (height, width) - model input size (same as img_info since we already rescaled)
        # Since our detections are already in original image coordinates,
        # we pass same size for both to avoid rescaling
        img_info = frame_shape
        img_size = frame_shape

        # OC-SORT update returns (M, 5) [x1, y1, x2, y2, track_id]
        online_targets = self.tracker.update(dets, img_info, img_size)

        if len(online_targets) == 0:
            return np.empty((0, 6))

        # Convert to output format (M, 6) [x1, y1, x2, y2, track_id, conf]
        # Simply use confidence 1.0 for all tracks (OC-SORT already filtered by det_thresh)
        tracks = np.zeros((len(online_targets), 6))
        tracks[:, :4] = online_targets[:, :4]  # x1, y1, x2, y2
        tracks[:, 4] = online_targets[:, 4].astype(int)  # track_id
        tracks[:, 5] = 1.0  # confidence (all tracks are confirmed)

        return tracks


    def reset(self):
        """Reset tracker state"""
        self.tracker = OCSort(
            det_thresh=self.det_thresh,
            max_age=self.max_age,
            min_hits=self.min_hits,
            iou_threshold=self.iou_threshold,
            delta_t=self.delta_t,
            asso_func=self.asso_func,
            inertia=self.inertia,
            use_byte=self.use_byte
        )
        self.frame_count = 0
        logger.info("OC-SORT Tracker reset")

    def get_info(self):
        """Get tracker information"""
        return {
            'det_thresh': self.det_thresh,
            'max_age': self.max_age,
            'min_hits': self.min_hits,
            'iou_threshold': self.iou_threshold,
            'delta_t': self.delta_t,
            'asso_func': self.asso_func,
            'inertia': self.inertia,
            'use_byte': self.use_byte,
            'frame_count': self.frame_count,
        }
