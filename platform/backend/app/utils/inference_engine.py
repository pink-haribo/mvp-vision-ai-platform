"""
Platform Inference Engine using ONNX Runtime.

Provides real-time inference for deployed models with automatic model loading,
preprocessing, and result formatting.
"""

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import tempfile
import shutil

import numpy as np
import onnxruntime as ort
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceEngine:
    """ONNX Runtime inference engine for platform deployments."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize inference engine.

        Args:
            cache_dir: Directory for caching downloaded models (default: temp dir)
        """
        self.cache_dir = cache_dir or Path(tempfile.gettempdir()) / "platform_inference_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Model cache: {deployment_id: (session, metadata)}
        self.model_cache: Dict[int, Tuple[ort.InferenceSession, Dict]] = {}

        logger.info(f"[InferenceEngine] Initialized with cache dir: {self.cache_dir}")

    def load_model(
        self,
        deployment_id: int,
        model_path: Path,
        metadata: Dict[str, Any]
    ) -> ort.InferenceSession:
        """
        Load ONNX model into memory.

        Args:
            deployment_id: Deployment ID for caching
            model_path: Path to ONNX model file
            metadata: Model metadata from export

        Returns:
            ONNX Runtime inference session
        """
        # Check cache
        if deployment_id in self.model_cache:
            logger.info(f"[InferenceEngine] Using cached model for deployment {deployment_id}")
            return self.model_cache[deployment_id][0]

        logger.info(f"[InferenceEngine] Loading model from: {model_path}")

        # Create ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Try GPU first, fall back to CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Cache model and metadata
        self.model_cache[deployment_id] = (session, metadata)

        logger.info(f"[InferenceEngine] Model loaded successfully")
        logger.info(f"[InferenceEngine] Providers: {session.get_providers()}")

        return session

    def preprocess_image(
        self,
        image_base64: str,
        metadata: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for inference.

        Args:
            image_base64: Base64 encoded image
            metadata: Model metadata with preprocessing specs

        Returns:
            preprocessed: Preprocessed image tensor (NCHW)
            info: Preprocessing info for inverse transform
        """
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        original_size = image.size  # (width, height)

        # Get target size from metadata
        input_shape = metadata.get('input_spec', {}).get('shape', [1, 3, 640, 640])
        target_h, target_w = input_shape[2], input_shape[3]

        # Letterbox resize
        preprocessed, ratio, padding = self._letterbox_resize(
            image, (target_w, target_h)
        )

        # Convert to numpy array
        preprocessed_np = np.array(preprocessed, dtype=np.float32)

        # Normalize
        preprocessing = metadata.get('preprocessing', {})
        mean = preprocessing.get('mean', [0.0, 0.0, 0.0])
        std = preprocessing.get('std', [255.0, 255.0, 255.0])

        preprocessed_np = (preprocessed_np - mean) / std

        # HWC to CHW
        preprocessed_np = preprocessed_np.transpose(2, 0, 1)

        # Add batch dimension
        preprocessed_np = preprocessed_np[np.newaxis, ...]

        info = {
            'original_size': original_size,
            'ratio': ratio,
            'padding': padding
        }

        return preprocessed_np, info

    def _letterbox_resize(
        self,
        image: Image.Image,
        target_size: Tuple[int, int]
    ) -> Tuple[Image.Image, float, Tuple[int, int]]:
        """
        Resize image with letterbox (maintain aspect ratio, add padding).

        Args:
            image: Input PIL image
            target_size: Target (width, height)

        Returns:
            resized_image: Letterboxed image
            ratio: Scale ratio
            padding: (pad_w, pad_h)
        """
        target_w, target_h = target_size
        original_w, original_h = image.size

        # Calculate scale ratio
        ratio = min(target_w / original_w, target_h / original_h)

        # Calculate new size
        new_w = int(original_w * ratio)
        new_h = int(original_h * ratio)

        # Resize
        resized = image.resize((new_w, new_h), Image.LANCZOS)

        # Calculate padding
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        # Create new image with padding
        padded = Image.new('RGB', target_size, (114, 114, 114))
        padded.paste(resized, (pad_w, pad_h))

        return padded, ratio, (pad_w, pad_h)

    def run_inference(
        self,
        session: ort.InferenceSession,
        input_tensor: np.ndarray
    ) -> List[np.ndarray]:
        """
        Run ONNX Runtime inference.

        Args:
            session: ONNX Runtime session
            input_tensor: Preprocessed input tensor

        Returns:
            List of output tensors
        """
        # Get input/output names
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]

        # Run inference
        outputs = session.run(output_names, {input_name: input_tensor})

        return outputs

    def postprocess_detection(
        self,
        outputs: List[np.ndarray],
        preprocess_info: Dict,
        metadata: Dict[str, Any],
        conf_threshold: float,
        iou_threshold: float,
        max_detections: int
    ) -> List[Dict[str, Any]]:
        """
        Postprocess detection outputs.

        Args:
            outputs: Raw model outputs
            preprocess_info: Preprocessing info for inverse transform
            metadata: Model metadata
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections

        Returns:
            List of detection dictionaries
        """
        # YOLOv8 output: [1, 84, 8400] -> [8400, 84]
        predictions = outputs[0][0].T  # [8400, 84]

        # Extract boxes and scores
        boxes = predictions[:, :4]  # [8400, 4] (cx, cy, w, h)
        class_scores = predictions[:, 4:]  # [8400, 80]

        scores = class_scores.max(axis=1)  # [8400]
        class_ids = class_scores.argmax(axis=1)  # [8400]

        # Filter by confidence
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert boxes from cxcywh to xyxy
        boxes = self._cxcywh_to_xyxy(boxes)

        # Scale boxes back to original image
        boxes = self._scale_boxes(boxes, preprocess_info)

        # Apply NMS
        keep_indices = self._nms(boxes, scores, iou_threshold)
        keep_indices = keep_indices[:max_detections]

        # Get class names
        class_names = metadata.get('model_info', {}).get('class_names', [])

        # Format detections
        detections = []
        for idx in keep_indices:
            detection = {
                'class_id': int(class_ids[idx]),
                'class_name': class_names[class_ids[idx]] if class_ids[idx] < len(class_names) else None,
                'confidence': float(scores[idx]),
                'bbox': {
                    'x1': float(boxes[idx, 0]),
                    'y1': float(boxes[idx, 1]),
                    'x2': float(boxes[idx, 2]),
                    'y2': float(boxes[idx, 3])
                }
            }
            detections.append(detection)

        return detections

    def _cxcywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from (cx, cy, w, h) to (x1, y1, x2, y2)."""
        xyxy = np.zeros_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
        return xyxy

    def _scale_boxes(self, boxes: np.ndarray, preprocess_info: Dict) -> np.ndarray:
        """Scale boxes from model input size to original image size."""
        ratio = preprocess_info['ratio']
        pad_w, pad_h = preprocess_info['padding']

        # Remove padding
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h

        # Scale
        boxes /= ratio

        # Clip to original image size
        original_w, original_h = preprocess_info['original_size']
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, original_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, original_h)

        return boxes

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return keep

    def infer(
        self,
        deployment_id: int,
        model_path: Path,
        metadata: Dict[str, Any],
        image_base64: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300
    ) -> Dict[str, Any]:
        """
        Run complete inference pipeline.

        Args:
            deployment_id: Deployment ID
            model_path: Path to ONNX model
            metadata: Model metadata
            image_base64: Base64 encoded image
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum detections

        Returns:
            Inference results dictionary
        """
        start_time = time.time()

        # Load model
        session = self.load_model(deployment_id, model_path, metadata)

        # Preprocess
        input_tensor, preprocess_info = self.preprocess_image(image_base64, metadata)

        # Run inference
        outputs = self.run_inference(session, input_tensor)

        # Postprocess
        task_type = metadata.get('model_info', {}).get('task_type', 'detect')

        if task_type in ['detect', 'detection', 'object_detection']:
            detections = self.postprocess_detection(
                outputs, preprocess_info, metadata,
                conf_threshold, iou_threshold, max_detections
            )
            result = {'detections': detections}
        else:
            # TODO: Add support for segmentation, pose, classification
            raise NotImplementedError(f"Task type {task_type} not yet supported")

        inference_time_ms = (time.time() - start_time) * 1000

        return {
            'task_type': task_type,
            'inference_time_ms': inference_time_ms,
            **result,
            'model_info': metadata.get('model_info', {})
        }

    def clear_cache(self, deployment_id: Optional[int] = None):
        """
        Clear model cache.

        Args:
            deployment_id: If provided, clear only this deployment. Otherwise clear all.
        """
        if deployment_id is not None:
            if deployment_id in self.model_cache:
                del self.model_cache[deployment_id]
                logger.info(f"[InferenceEngine] Cleared cache for deployment {deployment_id}")
        else:
            self.model_cache.clear()
            logger.info(f"[InferenceEngine] Cleared all model cache")


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def get_inference_engine() -> InferenceEngine:
    """Get global inference engine instance."""
    global _inference_engine
    if _inference_engine is None:
        _inference_engine = InferenceEngine()
    return _inference_engine
