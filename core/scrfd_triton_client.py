"""
SCRFD Face Detector Triton Client
High-performance face detection using Triton Inference Server
"""

import cv2
import numpy as np
from loguru import logger
import tritonclient.grpc as grpcclient


class SCRFDTritonClient:
    """SCRFD face detector client for Triton Inference Server"""
    
    def __init__(self, triton_url='localhost:8101', model_name='scrfd_10g', 
                 input_size=(640, 640), conf_threshold=0.5, nms_threshold=0.4):
        """
        Args:
            triton_url: Triton server gRPC URL
            model_name: Model name in Triton repository
            input_size: Input image size (height, width)
            conf_threshold: Confidence threshold for face detection
            nms_threshold: NMS IoU threshold
        """
        self.triton_url = triton_url
        self.model_name = model_name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        
        # SCRFD anchor settings
        self.fmc = 3  # Feature map count
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        
        # Initialize Triton client
        try:
            self.client = grpcclient.InferenceServerClient(url=triton_url)
            
            # Check server health
            if not self.client.is_server_live():
                raise RuntimeError(f"Triton server not live at {triton_url}")
            
            # Check model ready
            if not self.client.is_model_ready(model_name):
                raise RuntimeError(f"Model {model_name} not ready")
            
            logger.info(f"✅ SCRFD Triton client connected: {triton_url}/{model_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Triton: {e}")
            raise
    
    def preprocess(self, img):
        """
        Preprocess image for SCRFD
        Args:
            img: Input image (H, W, 3) BGR
        Returns:
            Preprocessed image (1, 3, H, W) FP32
        """
        # Resize to input size
        img_resized = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize: (img - 127.5) / 128.0
        img_normalized = (img_rgb.astype(np.float32) - 127.5) / 128.0
        
        # Transpose to (3, H, W)
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        
        # Add batch dimension (1, 3, H, W)
        img_batch = np.expand_dims(img_transposed, axis=0)
        
        return img_batch.astype(np.float32)
    
    def distance2bbox(self, points, distance, max_shape=None):
        """Decode distance prediction to bounding box"""
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def distance2kps(self, points, distance, max_shape=None):
        """Decode distance prediction to keypoints"""
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, 0] + distance[:, i]
            py = points[:, 1] + distance[:, i + 1]
            
            if max_shape is not None:
                px = np.clip(px, 0, max_shape[1])
                py = np.clip(py, 0, max_shape[0])
            
            preds.append(px)
            preds.append(py)
        
        return np.stack(preds, axis=-1)
    
    def postprocess(self, outputs, img_shape, input_shape):
        """
        Postprocess SCRFD outputs
        Args:
            outputs: List of output tensors from Triton
            img_shape: Original image shape (H, W)
            input_shape: Input shape to model (H, W)
        Returns:
            Detected faces: (N, 15) [x1, y1, x2, y2, conf, kp1_x, kp1_y, ..., kp5_x, kp5_y]
        """
        # Parse outputs: 3 scales x (score, bbox, kps)
        scores_list = []
        bboxes_list = []
        kpss_list = []

        # SCRFD outputs: 9 tensors organized by type (not by scale!)
        # Triton output order: [score_s8, score_s16, score_s32, bbox_s8, bbox_s16, bbox_s32, kps_s8, kps_s16, kps_s32]
        # We need to reorganize them by scale
        num_scales = len(self._feat_stride_fpn)

        # Calculate scale factor
        scale_x = img_shape[1] / input_shape[1]
        scale_y = img_shape[0] / input_shape[0]

        for idx, stride in enumerate(self._feat_stride_fpn):
            # Outputs are organized: [scores (3), bboxes (3), kps (3)]
            score = outputs[idx]  # scores: index 0, 1, 2
            bbox = outputs[idx + num_scales]  # bboxes: index 3, 4, 5
            kps = outputs[idx + num_scales * 2]  # kps: index 6, 7, 8

            # Remove batch dimension if present
            if score.ndim == 3:
                score = score[0]
            if bbox.ndim == 3:
                bbox = bbox[0]
            if kps.ndim == 3:
                kps = kps[0]

            # Generate anchor centers for this scale
            height = input_shape[0] // stride
            width = input_shape[1] // stride

            # Create anchor grid
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape(-1, 2)

            # Repeat for num_anchors (SCRFD uses 2 anchors per location)
            num_anchors = self._num_anchors
            anchor_centers = np.repeat(anchor_centers, num_anchors, axis=0)

            # Decode bbox from distance offsets
            bbox_decoded = self.distance2bbox(anchor_centers, bbox * stride)

            # Decode keypoints from distance offsets
            kps_decoded = self.distance2kps(anchor_centers, kps * stride)

            # Scale to original image size
            bbox_decoded[:, [0, 2]] *= scale_x
            bbox_decoded[:, [1, 3]] *= scale_y
            kps_decoded[:, 0::2] *= scale_x
            kps_decoded[:, 1::2] *= scale_y

            scores_list.append(score)
            bboxes_list.append(bbox_decoded)
            kpss_list.append(kps_decoded)

        # Concatenate all scales
        scores = np.vstack(scores_list)  # (total_anchors, 1)
        bboxes = np.vstack(bboxes_list)  # (total_anchors, 4)
        kpss = np.vstack(kpss_list)  # (total_anchors, 10)

        # Filter by confidence
        scores = scores.squeeze()
        mask = scores > self.conf_threshold

        if mask.sum() == 0:
            return np.empty((0, 15), dtype=np.float32)

        scores = scores[mask]
        bboxes = bboxes[mask]
        kpss = kpss[mask]

        # Combine: [x1, y1, x2, y2, conf, kp1_x, kp1_y, ..., kp5_x, kp5_y]
        detections = np.hstack([bboxes, scores[:, None], kpss])

        # Apply NMS
        detections = self.nms(detections)

        return detections

    def nms(self, dets):
        """
        Non-Maximum Suppression
        Args:
            dets: (N, 15) [x1, y1, x2, y2, conf, ...]
        Returns:
            Filtered detections
        """
        if len(dets) == 0:
            return dets

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]

        return dets[keep]

    def detect(self, img, timeout=0.1):
        """
        Detect faces in image
        Args:
            img: Input image (H, W, 3) BGR
            timeout: Inference timeout in seconds (default: 100ms)
        Returns:
            Faces: (N, 15) [x1, y1, x2, y2, conf, kp1_x, kp1_y, ..., kp5_x, kp5_y]
        """
        img_shape = img.shape[:2]

        # Preprocess
        img_preprocessed = self.preprocess(img)

        # Create Triton input
        inputs = [
            grpcclient.InferInput('input.1', img_preprocessed.shape, 'FP32')
        ]
        inputs[0].set_data_from_numpy(img_preprocessed)

        # Create Triton outputs (9 outputs for 3 scales)
        output_names = ['448', '471', '494', '451', '474', '497', '454', '477', '500']
        outputs = [grpcclient.InferRequestedOutput(name) for name in output_names]

        # Inference
        try:
            response = self.client.infer(
                model_name=self.model_name,
                inputs=inputs,
                outputs=outputs,
                timeout=int(timeout * 1e6)  # Convert to microseconds
            )

            # Get outputs
            output_tensors = [response.as_numpy(name) for name in output_names]

            # Postprocess
            faces = self.postprocess(output_tensors, img_shape, self.input_size)

            return faces

        except Exception as e:
            logger.warning(f"SCRFD Triton inference failed: {e}")
            return np.empty((0, 15), dtype=np.float32)

    def detect_faces_in_crop(self, crop):
        """
        Detect faces in a cropped person region
        Args:
            crop: Cropped person image (H, W, 3) BGR
        Returns:
            List of face bboxes relative to crop: [[x1, y1, x2, y2, conf], ...]
        """
        if crop.size == 0:
            return []

        # Detect faces
        faces = self.detect(crop)

        if len(faces) == 0:
            return []

        # Extract bboxes and confidence
        face_bboxes = []
        for face in faces:
            x1, y1, x2, y2, conf = face[:5]
            face_bboxes.append([x1, y1, x2, y2, conf])

        return face_bboxes
