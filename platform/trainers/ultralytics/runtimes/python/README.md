# Python ONNX Runtime Wrapper

Easy-to-use Python wrapper for running YOLO models exported to ONNX format.

## Installation

```bash
pip install -r requirements.txt
```

For GPU acceleration (optional):
```bash
pip install onnxruntime-gpu
```

## Quick Start

```python
from model_wrapper import YOLOInference

# Initialize model
model = YOLOInference("model.onnx", metadata_path="metadata.json")

# Run inference
results = model.predict("image.jpg", conf_threshold=0.25, iou_threshold=0.45)

# Visualize results
annotated = model.visualize(results, "image.jpg")

# Save output
import cv2
cv2.imwrite("output.jpg", annotated)
```

## Supported Tasks

- **Detection**: Object detection with bounding boxes
- **Segmentation**: Instance segmentation with masks
- **Pose Estimation**: Human pose keypoints (17 keypoints)
- **Classification**: Image classification

## API Reference

### YOLOInference

```python
model = YOLOInference(
    model_path="model.onnx",
    metadata_path="metadata.json",  # Optional, auto-detected
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']  # Optional
)
```

**Parameters:**
- `model_path`: Path to ONNX model file
- `metadata_path`: Path to metadata.json (default: auto-detect in same folder)
- `providers`: ONNX Runtime execution providers (default: CUDA + CPU)

### predict()

```python
results = model.predict(
    image="image.jpg",
    conf_threshold=0.25,
    iou_threshold=0.45,
    max_det=300
)
```

**Parameters:**
- `image`: Image path (str) or numpy array (BGR format)
- `conf_threshold`: Confidence threshold (0.0-1.0)
- `iou_threshold`: IoU threshold for NMS (0.0-1.0)
- `max_det`: Maximum number of detections

**Returns:**

For **detection**:
```python
{
    'boxes': np.ndarray,   # [N, 4] in xyxy format
    'scores': np.ndarray,  # [N]
    'classes': np.ndarray  # [N]
}
```

For **segmentation**:
```python
{
    'boxes': np.ndarray,   # [N, 4]
    'scores': np.ndarray,  # [N]
    'classes': np.ndarray, # [N]
    'masks': np.ndarray    # [N, H, W]
}
```

For **pose**:
```python
{
    'boxes': np.ndarray,      # [N, 4]
    'scores': np.ndarray,     # [N]
    'keypoints': np.ndarray,  # [N, 17, 3] (x, y, confidence)
    'classes': np.ndarray     # [N] (always 0 for pose)
}
```

For **classification**:
```python
{
    'class': int,               # Predicted class ID
    'score': float,             # Confidence score
    'probabilities': np.ndarray # [num_classes]
}
```

### visualize()

```python
annotated = model.visualize(
    results=results,
    image="image.jpg",
    show_labels=True,
    show_conf=True,
    line_width=None  # Auto
)
```

**Parameters:**
- `results`: Prediction results from `predict()`
- `image`: Image path or numpy array
- `show_labels`: Show class labels
- `show_conf`: Show confidence scores
- `line_width`: Bounding box line width (auto if None)

**Returns:**
- `annotated_image`: np.ndarray with drawn predictions

## Command Line Usage

```bash
python model_wrapper.py model.onnx image.jpg
# Output saved to: output.jpg
```

## Advanced Usage

### Batch Inference

```python
import cv2
from pathlib import Path

model = YOLOInference("model.onnx")

for image_path in Path("images").glob("*.jpg"):
    image = cv2.imread(str(image_path))
    results = model.predict(image, conf_threshold=0.3)

    annotated = model.visualize(results, image)
    cv2.imwrite(f"output/{image_path.name}", annotated)
```

### Custom Preprocessing

```python
# Access preprocessing info
input_tensor, preprocess_info = model.preprocess(image)
print(f"Original shape: {preprocess_info['original_shape']}")
print(f"Scale ratio: {preprocess_info['ratio']}")
print(f"Padding: {preprocess_info['pad']}")
```

### GPU Acceleration

```python
# Force GPU execution
model = YOLOInference(
    "model.onnx",
    providers=['CUDAExecutionProvider']
)

# Check active provider
print(f"Using: {model.session.get_providers()}")
```

### Video Inference

```python
import cv2

model = YOLOInference("model.onnx")
cap = cv2.VideoCapture("video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf_threshold=0.25)
    annotated = model.visualize(results, frame)

    cv2.imshow("YOLO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Performance Tips

1. **Use GPU**: Install `onnxruntime-gpu` for ~10x speedup
2. **Batch processing**: Process multiple images in parallel
3. **Lower resolution**: Use smaller input size for faster inference
4. **Increase thresholds**: Higher `conf_threshold` = fewer boxes = faster NMS

## Troubleshooting

**Q: "CUDAExecutionProvider not available"**
```bash
pip install onnxruntime-gpu
# Make sure CUDA is installed
```

**Q: "Model input shape mismatch"**
- Check metadata.json has correct input_shape
- Ensure image preprocessing matches training

**Q: "No detections returned"**
- Lower `conf_threshold` (try 0.1)
- Check if model was trained properly
- Verify class names in metadata.json

## License

See parent LICENSE file.
