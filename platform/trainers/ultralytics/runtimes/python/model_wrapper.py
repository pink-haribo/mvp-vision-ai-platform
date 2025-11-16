"""
ONNX Runtime wrapper for YOLO models exported from Vision AI Training Platform.

This wrapper provides easy-to-use inference with preprocessing and postprocessing
for YOLO models exported to ONNX format.

Supports:
- YOLOv8/YOLO11 Detection
- YOLOv8/YOLO11 Segmentation
- YOLOv8/YOLO11 Pose Estimation
- YOLOv8/YOLO11 Classification

Requirements:
    pip install onnxruntime numpy opencv-python

Usage:
    from model_wrapper import YOLOInference

    # Initialize
    model = YOLOInference("model.onnx", metadata_path="metadata.json")

    # Run inference
    results = model.predict("image.jpg", conf_threshold=0.25, iou_threshold=0.45)

    # Visualize
    annotated = model.visualize(results, "image.jpg")
    cv2.imwrite("output.jpg", annotated)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YOLOInference:
    """ONNX Runtime inference wrapper for YOLO models."""

    def __init__(
        self,
        model_path: Union[str, Path],
        metadata_path: Optional[Union[str, Path]] = None,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize YOLO inference engine.

        Args:
            model_path: Path to ONNX model file
            metadata_path: Path to metadata.json (optional, auto-detected if None)
            providers: ONNX Runtime execution providers
                      Default: ['CUDAExecutionProvider', 'CPUExecutionProvider']
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Auto-detect metadata.json
        if metadata_path is None:
            metadata_path = self.model_path.parent / "metadata.json"

        self.metadata_path = Path(metadata_path)
        self.metadata = self._load_metadata()

        # Initialize ONNX Runtime session
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers
        )

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]

        logger.info(f"Model loaded: {self.model_path.name}")
        logger.info(f"Task: {self.metadata.get('task_type', 'unknown')}")
        logger.info(f"Input shape: {self.input_shape}")
        logger.info(f"Providers: {self.session.get_providers()}")

    def _load_metadata(self) -> Dict:
        """Load metadata.json."""
        if not self.metadata_path.exists():
            logger.warning(f"Metadata not found: {self.metadata_path}, using defaults")
            return {
                "task_type": "detect",
                "input_shape": [1, 3, 640, 640],
                "class_names": [],
                "preprocessing": {
                    "resize": "letterbox",
                    "normalize": True,
                    "mean": [0.0, 0.0, 0.0],
                    "std": [255.0, 255.0, 255.0]
                }
            }

        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for YOLO inference.

        Args:
            image: Input image (BGR format, HWC)

        Returns:
            preprocessed: Preprocessed image tensor (NCHW)
            info: Preprocessing info (scale, padding) for postprocessing
        """
        original_shape = image.shape[:2]  # (height, width)

        # Get target size from metadata or input shape
        input_shape = self.metadata.get('input_shape', self.input_shape)
        target_h, target_w = input_shape[2], input_shape[3]

        # Letterbox resize (maintain aspect ratio)
        preprocessed, ratio, (pad_w, pad_h) = self._letterbox_resize(
            image, (target_w, target_h)
        )

        # BGR to RGB
        preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

        # Normalize
        preprocessed = preprocessed.astype(np.float32)
        mean = self.metadata.get('preprocessing', {}).get('mean', [0.0, 0.0, 0.0])
        std = self.metadata.get('preprocessing', {}).get('std', [255.0, 255.0, 255.0])

        preprocessed = (preprocessed - mean) / std

        # HWC to CHW
        preprocessed = preprocessed.transpose(2, 0, 1)

        # Add batch dimension
        preprocessed = preprocessed[np.newaxis, ...]

        info = {
            'original_shape': original_shape,
            'ratio': ratio,
            'pad': (pad_w, pad_h)
        }

        return preprocessed, info

    def _letterbox_resize(
        self,
        image: np.ndarray,
        new_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterbox (maintain aspect ratio, add padding).

        Args:
            image: Input image
            new_shape: Target (width, height)

        Returns:
            resized_image: Letterboxed image
            ratio: Scale ratio
            padding: (pad_w, pad_h)
        """
        shape = image.shape[:2]  # current (height, width)
        new_w, new_h = new_shape

        # Scale ratio (new / old)
        r = min(new_h / shape[0], new_w / shape[1])

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_w - new_unpad[0], new_h - new_unpad[1]

        dw /= 2
        dh /= 2

        # Resize
        if shape[::-1] != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        # Add border
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(
            image, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return image, r, (left, top)

    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_det: int = 300
    ) -> Dict:
        """
        Run inference on image.

        Args:
            image: Image path or numpy array (BGR)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections

        Returns:
            results: Dictionary with predictions
                - boxes: [N, 4] xyxy format
                - scores: [N]
                - classes: [N]
                - masks: [N, H, W] (if segmentation)
                - keypoints: [N, K, 3] (if pose)
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
            if image is None:
                raise ValueError(f"Failed to load image: {image}")

        # Preprocess
        input_tensor, preprocess_info = self.preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # Postprocess
        task_type = self.metadata.get('task_type', 'detect')

        if task_type in ['detect', 'detection']:
            results = self._postprocess_detect(
                outputs, preprocess_info, conf_threshold, iou_threshold, max_det
            )
        elif task_type in ['segment', 'segmentation']:
            results = self._postprocess_segment(
                outputs, preprocess_info, conf_threshold, iou_threshold, max_det
            )
        elif task_type in ['pose', 'pose_estimation']:
            results = self._postprocess_pose(
                outputs, preprocess_info, conf_threshold, iou_threshold, max_det
            )
        elif task_type in ['classify', 'classification']:
            results = self._postprocess_classify(outputs, conf_threshold)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        return results

    def _postprocess_detect(
        self,
        outputs: List[np.ndarray],
        preprocess_info: Dict,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int
    ) -> Dict:
        """Postprocess detection outputs."""
        predictions = outputs[0]  # [1, 84, 8400] for YOLOv8
        predictions = predictions[0]  # [84, 8400]
        predictions = predictions.T  # [8400, 84]

        # Extract boxes and scores
        boxes = predictions[:, :4]  # [8400, 4] (cx, cy, w, h)
        scores = predictions[:, 4:].max(axis=1)  # [8400]
        class_ids = predictions[:, 4:].argmax(axis=1)  # [8400]

        # Filter by confidence
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return {'boxes': np.array([]), 'scores': np.array([]), 'classes': np.array([])}

        # Convert boxes from cxcywh to xyxy
        boxes = self._cxcywh_to_xyxy(boxes)

        # Scale boxes back to original image
        boxes = self._scale_boxes(boxes, preprocess_info)

        # Apply NMS
        keep_indices = self._nms(boxes, scores, iou_threshold)
        keep_indices = keep_indices[:max_det]

        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'classes': class_ids[keep_indices]
        }

    def _postprocess_segment(
        self,
        outputs: List[np.ndarray],
        preprocess_info: Dict,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int
    ) -> Dict:
        """Postprocess segmentation outputs."""
        # Detection head
        predictions = outputs[0][0].T  # [8400, 116]

        # Mask protos
        protos = outputs[1][0]  # [32, 160, 160]

        # Extract boxes, scores, masks
        boxes = predictions[:, :4]
        scores = predictions[:, 4:84].max(axis=1)
        class_ids = predictions[:, 4:84].argmax(axis=1)
        mask_coeffs = predictions[:, 84:]  # [8400, 32]

        # Filter by confidence
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]
        mask_coeffs = mask_coeffs[mask]

        if len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'classes': np.array([]),
                'masks': np.array([])
            }

        # Convert and scale boxes
        boxes = self._cxcywh_to_xyxy(boxes)
        boxes = self._scale_boxes(boxes, preprocess_info)

        # Apply NMS
        keep_indices = self._nms(boxes, scores, iou_threshold)
        keep_indices = keep_indices[:max_det]

        # Generate masks
        masks = self._generate_masks(
            mask_coeffs[keep_indices],
            protos,
            boxes[keep_indices],
            preprocess_info['original_shape']
        )

        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'classes': class_ids[keep_indices],
            'masks': masks
        }

    def _postprocess_pose(
        self,
        outputs: List[np.ndarray],
        preprocess_info: Dict,
        conf_threshold: float,
        iou_threshold: float,
        max_det: int
    ) -> Dict:
        """Postprocess pose estimation outputs."""
        predictions = outputs[0][0].T  # [8400, 56] for 17 keypoints

        # Extract boxes, scores, keypoints
        boxes = predictions[:, :4]
        scores = predictions[:, 4]
        keypoints = predictions[:, 5:].reshape(-1, 17, 3)  # [8400, 17, 3]

        # Filter by confidence
        mask = scores > conf_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        keypoints = keypoints[mask]

        if len(boxes) == 0:
            return {
                'boxes': np.array([]),
                'scores': np.array([]),
                'keypoints': np.array([])
            }

        # Convert and scale boxes
        boxes = self._cxcywh_to_xyxy(boxes)
        boxes = self._scale_boxes(boxes, preprocess_info)

        # Scale keypoints
        keypoints = self._scale_keypoints(keypoints, preprocess_info)

        # Apply NMS
        keep_indices = self._nms(boxes, scores, iou_threshold)
        keep_indices = keep_indices[:max_det]

        return {
            'boxes': boxes[keep_indices],
            'scores': scores[keep_indices],
            'keypoints': keypoints[keep_indices],
            'classes': np.zeros(len(keep_indices), dtype=int)  # Pose is single class
        }

    def _postprocess_classify(self, outputs: List[np.ndarray], conf_threshold: float) -> Dict:
        """Postprocess classification outputs."""
        logits = outputs[0][0]  # [num_classes]
        probs = self._softmax(logits)

        top_idx = np.argmax(probs)
        top_prob = probs[top_idx]

        if top_prob < conf_threshold:
            return {'class': -1, 'score': 0.0, 'probabilities': probs}

        return {
            'class': int(top_idx),
            'score': float(top_prob),
            'probabilities': probs
        }

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
        pad_w, pad_h = preprocess_info['pad']

        # Remove padding
        boxes[:, [0, 2]] -= pad_w
        boxes[:, [1, 3]] -= pad_h

        # Scale
        boxes /= ratio

        # Clip to original image size
        h, w = preprocess_info['original_shape']
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)

        return boxes

    def _scale_keypoints(self, keypoints: np.ndarray, preprocess_info: Dict) -> np.ndarray:
        """Scale keypoints from model input size to original image size."""
        ratio = preprocess_info['ratio']
        pad_w, pad_h = preprocess_info['pad']

        keypoints[..., 0] = (keypoints[..., 0] - pad_w) / ratio
        keypoints[..., 1] = (keypoints[..., 1] - pad_h) / ratio

        return keypoints

    def _generate_masks(
        self,
        mask_coeffs: np.ndarray,
        protos: np.ndarray,
        boxes: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Generate instance masks from coefficients and prototypes."""
        # Matrix multiplication: [N, 32] @ [32, 160, 160] -> [N, 160, 160]
        masks = mask_coeffs @ protos.reshape(32, -1)
        masks = masks.reshape(-1, protos.shape[1], protos.shape[2])

        # Sigmoid activation
        masks = 1 / (1 + np.exp(-masks))

        # Resize to original image size
        masks_resized = []
        for mask, box in zip(masks, boxes):
            # Crop to bounding box
            x1, y1, x2, y2 = box.astype(int)
            mask_crop = cv2.resize(mask, (x2 - x1, y2 - y1))

            # Create full-size mask
            full_mask = np.zeros(original_shape, dtype=np.float32)
            full_mask[y1:y2, x1:x2] = mask_crop
            masks_resized.append(full_mask)

        return np.array(masks_resized)

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
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

        return np.array(keep)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax activation."""
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()

    def visualize(
        self,
        results: Dict,
        image: Union[str, Path, np.ndarray],
        show_labels: bool = True,
        show_conf: bool = True,
        line_width: Optional[int] = None
    ) -> np.ndarray:
        """
        Visualize predictions on image.

        Args:
            results: Prediction results from predict()
            image: Image path or numpy array
            show_labels: Show class labels
            show_conf: Show confidence scores
            line_width: Bounding box line width (auto if None)

        Returns:
            annotated_image: Image with annotations
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = cv2.imread(str(image))
        else:
            image = image.copy()

        task_type = self.metadata.get('task_type', 'detect')
        class_names = self.metadata.get('class_names', [])

        if line_width is None:
            line_width = max(round(sum(image.shape) / 2 * 0.003), 2)

        if task_type in ['detect', 'detection']:
            image = self._draw_boxes(
                image, results, class_names, show_labels, show_conf, line_width
            )
        elif task_type in ['segment', 'segmentation']:
            image = self._draw_masks(image, results, class_names, show_labels, show_conf)
        elif task_type in ['pose', 'pose_estimation']:
            image = self._draw_pose(image, results, show_conf, line_width)
        elif task_type in ['classify', 'classification']:
            image = self._draw_classification(image, results, class_names)

        return image

    def _draw_boxes(
        self,
        image: np.ndarray,
        results: Dict,
        class_names: List[str],
        show_labels: bool,
        show_conf: bool,
        line_width: int
    ) -> np.ndarray:
        """Draw bounding boxes."""
        boxes = results.get('boxes', [])
        scores = results.get('scores', [])
        classes = results.get('classes', [])

        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)

            # Random color based on class
            color = self._get_color(int(cls))

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, line_width)

            # Draw label
            if show_labels or show_conf:
                label = ""
                if show_labels and cls < len(class_names):
                    label = class_names[int(cls)]
                if show_conf:
                    label += f" {score:.2f}" if label else f"{score:.2f}"

                # Text background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), color, -1)
                cv2.putText(
                    image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
                )

        return image

    def _draw_masks(
        self,
        image: np.ndarray,
        results: Dict,
        class_names: List[str],
        show_labels: bool,
        show_conf: bool
    ) -> np.ndarray:
        """Draw segmentation masks."""
        boxes = results.get('boxes', [])
        scores = results.get('scores', [])
        classes = results.get('classes', [])
        masks = results.get('masks', [])

        # Create overlay
        overlay = image.copy()

        for box, score, cls, mask in zip(boxes, scores, classes, masks):
            color = self._get_color(int(cls))

            # Draw mask
            overlay[mask > 0.5] = overlay[mask > 0.5] * 0.5 + np.array(color) * 0.5

            # Draw box and label
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            if show_labels or show_conf:
                label = ""
                if show_labels and cls < len(class_names):
                    label = class_names[int(cls)]
                if show_conf:
                    label += f" {score:.2f}" if label else f"{score:.2f}"

                cv2.putText(
                    overlay, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

        return overlay

    def _draw_pose(
        self,
        image: np.ndarray,
        results: Dict,
        show_conf: bool,
        line_width: int
    ) -> np.ndarray:
        """Draw pose keypoints and skeleton."""
        boxes = results.get('boxes', [])
        scores = results.get('scores', [])
        keypoints = results.get('keypoints', [])

        # COCO skeleton
        skeleton = [
            [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
            [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
            [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]
        ]

        for box, score, kpts in zip(boxes, scores, keypoints):
            # Draw skeleton
            for sk in skeleton:
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))

                conf1 = kpts[sk[0] - 1, 2]
                conf2 = kpts[sk[1] - 1, 2]

                if conf1 > 0.5 and conf2 > 0.5:
                    cv2.line(image, pos1, pos2, (0, 255, 0), line_width)

            # Draw keypoints
            for i, kpt in enumerate(kpts):
                x, y, conf = kpt
                if conf > 0.5:
                    cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)

            # Draw box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if show_conf:
                cv2.putText(
                    image, f"{score:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
                )

        return image

    def _draw_classification(
        self,
        image: np.ndarray,
        results: Dict,
        class_names: List[str]
    ) -> np.ndarray:
        """Draw classification result."""
        cls = results.get('class', -1)
        score = results.get('score', 0.0)

        if cls >= 0 and cls < len(class_names):
            label = f"{class_names[cls]}: {score:.3f}"
        else:
            label = f"Class {cls}: {score:.3f}"

        # Draw label at top
        cv2.putText(
            image, label, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2
        )

        return image

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get consistent color for class ID."""
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())


# Example usage
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python model_wrapper.py <model.onnx> <image.jpg>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]

    # Initialize
    model = YOLOInference(model_path)

    # Run inference
    results = model.predict(image_path, conf_threshold=0.25, iou_threshold=0.45)

    print(f"Detections: {len(results.get('boxes', []))}")

    # Visualize
    annotated = model.visualize(results, image_path)
    output_path = "output.jpg"
    cv2.imwrite(output_path, annotated)

    print(f"Saved to: {output_path}")
