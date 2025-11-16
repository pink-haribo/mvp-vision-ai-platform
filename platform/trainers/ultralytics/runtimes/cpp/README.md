# C++ ONNX Runtime Wrapper

High-performance C++ wrapper for running YOLO models exported to ONNX format.

## Requirements

- C++17 compiler (GCC 7+, Clang 5+, MSVC 2019+)
- CMake 3.15+
- OpenCV 4.x
- ONNX Runtime 1.16+
- nlohmann/json (header-only, included in project)

## Installation

### 1. Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get install cmake g++ libopencv-dev
```

**macOS:**
```bash
brew install cmake opencv
```

**Windows:**
- Install Visual Studio 2019+
- Install OpenCV from https://opencv.org/releases/
- Install CMake from https://cmake.org/download/

### 2. Download ONNX Runtime

Download prebuilt binaries from https://github.com/microsoft/onnxruntime/releases

```bash
# Linux
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz

# macOS
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-osx-arm64-1.16.3.tgz
tar -xzf onnxruntime-osx-arm64-1.16.3.tgz

# Windows
# Download and extract onnxruntime-win-x64-1.16.3.zip
```

### 3. Build

```bash
mkdir build && cd build

cmake .. -DONNXRUNTIME_DIR=/path/to/onnxruntime
make -j$(nproc)
```

**Windows:**
```cmd
mkdir build && cd build
cmake .. -DONNXRUNTIME_DIR=C:\path\to\onnxruntime -G "Visual Studio 16 2019"
cmake --build . --config Release
```

## Quick Start

```cpp
#include "model_wrapper.h"

int main() {
    // Initialize model
    yolo::YOLOInference model("model.onnx", "metadata.json", true);

    // Load image
    cv::Mat image = cv::imread("image.jpg");

    // Run inference
    auto detections = model.predict(image, 0.25f, 0.45f);

    // Visualize
    cv::Mat annotated = model.visualize(detections, image);
    cv::imwrite("output.jpg", annotated);

    return 0;
}
```

**Compile:**
```bash
g++ -std=c++17 -O3 main.cpp model_wrapper.cpp \
    -I/path/to/onnxruntime/include \
    -L/path/to/onnxruntime/lib -lonnxruntime \
    `pkg-config --cflags --libs opencv4` \
    -o yolo_inference
```

**Run:**
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
./yolo_inference model.onnx image.jpg
```

## API Reference

### YOLOInference

```cpp
yolo::YOLOInference model(
    const std::string& modelPath,      // Path to ONNX model
    const std::string& metadataPath,   // Path to metadata.json (optional)
    bool useGPU = true                 // Enable GPU acceleration
);
```

### predict()

```cpp
std::vector<Detection> detections = model.predict(
    const cv::Mat& image,              // Input image (BGR)
    float confThreshold = 0.25f,       // Confidence threshold
    float iouThreshold = 0.45f,        // IoU threshold for NMS
    int maxDet = 300                   // Maximum detections
);
```

**Detection structure:**
```cpp
struct Detection {
    cv::Rect2f box;        // Bounding box (x, y, width, height)
    float confidence;      // Confidence score
    int classId;          // Class ID
    cv::Mat mask;         // Instance mask (segmentation only)
};
```

### visualize()

```cpp
cv::Mat annotated = model.visualize(
    const std::vector<Detection>& detections,
    const cv::Mat& image,
    bool showLabels = true,
    bool showConf = true
);
```

## Advanced Usage

### Batch Inference

```cpp
#include <filesystem>
namespace fs = std::filesystem;

yolo::YOLOInference model("model.onnx");

for (const auto& entry : fs::directory_iterator("images")) {
    cv::Mat image = cv::imread(entry.path());
    auto detections = model.predict(image, 0.3f, 0.45f);

    cv::Mat annotated = model.visualize(detections, image);
    cv::imwrite("output/" + entry.path().filename(), annotated);
}
```

### Video Inference

```cpp
yolo::YOLOInference model("model.onnx");
cv::VideoCapture cap("video.mp4");

cv::Mat frame;
while (cap.read(frame)) {
    auto detections = model.predict(frame, 0.25f, 0.45f);
    cv::Mat annotated = model.visualize(detections, frame);

    cv::imshow("YOLO", annotated);
    if (cv::waitKey(1) == 'q') break;
}
```

### GPU Acceleration

```cpp
// Force GPU execution
yolo::YOLOInference model("model.onnx", "", true);

// CPU only
yolo::YOLOInference model("model.onnx", "", false);
```

### Custom Postprocessing

```cpp
yolo::YOLOInference model("model.onnx");
cv::Mat image = cv::imread("image.jpg");

auto detections = model.predict(image, 0.25f, 0.45f);

// Filter by class
std::vector<yolo::Detection> personDetections;
for (const auto& det : detections) {
    if (det.classId == 0) {  // Person class in COCO
        personDetections.push_back(det);
    }
}

// Custom visualization
cv::Mat annotated = image.clone();
for (const auto& det : personDetections) {
    cv::rectangle(annotated, det.box, cv::Scalar(0, 255, 0), 2);
    std::string label = cv::format("%.2f", det.confidence);
    cv::putText(annotated, label,
                cv::Point(det.box.x, det.box.y - 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 255, 0), 2);
}
```

## Performance Optimization

1. **Use GPU**: ~10-20x faster than CPU
   ```cpp
   yolo::YOLOInference model("model.onnx", "", true);
   ```

2. **Optimize model**: Use TensorRT for NVIDIA GPUs
   ```bash
   # Export to TensorRT instead of ONNX
   # Use model.engine instead
   ```

3. **Image preprocessing**: Reuse buffers
   ```cpp
   cv::Mat buffer;
   for (const auto& image : images) {
       cv::resize(image, buffer, cv::Size(640, 640));
       auto detections = model.predict(buffer);
   }
   ```

4. **Multithreading**: Process multiple images in parallel
   ```cpp
   #pragma omp parallel for
   for (int i = 0; i < images.size(); i++) {
       auto detections = model.predict(images[i]);
   }
   ```

## Troubleshooting

**Q: "libonnxruntime.so: cannot open shared object file"**
```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
# Or add to ~/.bashrc
```

**Q: "CUDAExecutionProvider failed"**
- Install CUDA Toolkit
- Use CPU version: `yolo::YOLOInference model("model.onnx", "", false);`

**Q: "OpenCV not found"**
```bash
# Ubuntu
sudo apt-get install libopencv-dev

# macOS
brew install opencv

# Or specify OpenCV path
cmake .. -DOpenCV_DIR=/path/to/opencv
```

**Q: Compilation errors with ONNX Runtime**
- Ensure correct ONNX Runtime version (1.16+)
- Check include path: `-I/path/to/onnxruntime/include`
- Check library path: `-L/path/to/onnxruntime/lib`

## Example Projects

See `main.cpp` for complete example.

## License

See parent LICENSE file.
