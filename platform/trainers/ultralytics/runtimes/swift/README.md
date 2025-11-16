# Swift CoreML Wrapper

Easy-to-use Swift wrapper for running YOLO models on iOS and macOS using CoreML.

## Requirements

- iOS 13+ / macOS 10.15+
- Xcode 12+
- Swift 5.5+
- CoreML framework
- Vision framework

## Installation

### Swift Package Manager

Add to your `Package.swift`:

```swift
dependencies: [
    .package(url: "https://github.com/your-repo/YOLOInference", from: "1.0.0")
]
```

### Manual Integration

1. Copy `ModelWrapper.swift` to your Xcode project
2. Drag and drop your `.mlmodel` or `.mlpackage` file into Xcode
3. Ensure `metadata.json` is in the same directory as the model

## Quick Start

```swift
import UIKit

// Initialize model
let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
let model = try YOLOInference(modelURL: modelURL)

// Load image
let image = UIImage(named: "test.jpg")!

// Run inference
let detections = try model.predict(
    image: image,
    confThreshold: 0.25,
    iouThreshold: 0.45
)

print("Detections: \(detections.count)")

// Visualize
let annotated = model.visualize(results: detections, image: image)
imageView.image = annotated
```

## API Reference

### YOLOInference

```swift
let model = try YOLOInference(
    modelURL: URL,          // URL to .mlmodel or .mlpackage
    metadataURL: URL? = nil // Optional metadata.json URL
)
```

### predict()

```swift
let detections = try model.predict(
    image: UIImage,
    confThreshold: Float = 0.25,
    iouThreshold: Float = 0.45,
    maxDet: Int = 300
)
```

**Detection structure:**
```swift
struct Detection {
    let box: CGRect        // Bounding box
    let confidence: Float  // Confidence score
    let classId: Int      // Class ID
    let className: String? // Class name (from metadata)
}
```

### visualize()

```swift
let annotated = model.visualize(
    results: [Detection],
    image: UIImage,
    showLabels: Bool = true,
    showConf: Bool = true
)
```

## Advanced Usage

### Camera Integration (iOS)

```swift
import AVFoundation

class CameraViewController: UIViewController {
    var model: YOLOInference!
    var captureSession: AVCaptureSession!

    override func viewDidLoad() {
        super.viewDidLoad()

        // Initialize model
        let modelURL = Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
        model = try! YOLOInference(modelURL: modelURL)

        // Setup camera
        setupCamera()
    }

    func setupCamera() {
        captureSession = AVCaptureSession()
        captureSession.sessionPreset = .hd1280x720

        guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
            return
        }

        let input = try! AVCaptureDeviceInput(device: camera)
        captureSession.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera"))
        captureSession.addOutput(output)

        captureSession.startRunning()
    }
}

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            return
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
        let uiImage = UIImage(cgImage: cgImage)

        // Run inference
        do {
            let detections = try model.predict(image: uiImage, confThreshold: 0.3)

            DispatchQueue.main.async {
                let annotated = self.model.visualize(results: detections, image: uiImage)
                self.imageView.image = annotated
            }
        } catch {
            print("Inference error: \(error)")
        }
    }
}
```

### Video Processing

```swift
import AVKit

func processVideo(videoURL: URL) {
    let asset = AVAsset(url: videoURL)
    let reader = try! AVAssetReader(asset: asset)

    let videoTrack = asset.tracks(withMediaType: .video).first!
    let output = AVAssetReaderTrackOutput(
        track: videoTrack,
        outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
    )

    reader.add(output)
    reader.startReading()

    while let sampleBuffer = output.copyNextSampleBuffer() {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            continue
        }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        let cgImage = context.createCGImage(ciImage, from: ciImage.extent)!
        let uiImage = UIImage(cgImage: cgImage)

        let detections = try! model.predict(image: uiImage)
        print("Frame detections: \(detections.count)")
    }
}
```

### Batch Processing

```swift
func processBatch(images: [UIImage]) async {
    await withTaskGroup(of: [Detection].self) { group in
        for image in images {
            group.addTask {
                return try! self.model.predict(image: image)
            }
        }

        for await detections in group {
            print("Detections: \(detections.count)")
        }
    }
}
```

### Custom Postprocessing

```swift
// Filter detections by class
let personDetections = detections.filter { $0.classId == 0 }

// Filter by confidence
let highConfDetections = detections.filter { $0.confidence > 0.5 }

// Filter by area
let largeDetections = detections.filter { detection in
    let area = detection.box.width * detection.box.height
    return area > 10000
}

// Custom visualization
func customVisualize(detections: [Detection], image: UIImage) -> UIImage {
    UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
    defer { UIGraphicsEndImageContext() }

    image.draw(at: .zero)

    let context = UIGraphicsGetCurrentContext()!

    for detection in detections where detection.classId == 0 {
        context.setStrokeColor(UIColor.green.cgColor)
        context.setLineWidth(4.0)
        context.stroke(detection.box)

        let label = String(format: "Person %.2f", detection.confidence)
        let attributes: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 18, weight: .bold),
            .foregroundColor: UIColor.white
        ]

        (label as NSString).draw(
            at: CGPoint(x: detection.box.minX, y: detection.box.minY - 25),
            withAttributes: attributes
        )
    }

    return UIGraphicsGetImageFromCurrentImageContext()!
}
```

### SwiftUI Integration

```swift
import SwiftUI

struct DetectionView: View {
    @State private var image: UIImage?
    @State private var annotatedImage: UIImage?
    @State private var detections: [Detection] = []

    let model = try! YOLOInference(
        modelURL: Bundle.main.url(forResource: "model", withExtension: "mlmodel")!
    )

    var body: some View {
        VStack {
            if let annotatedImage = annotatedImage {
                Image(uiImage: annotatedImage)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
            }

            Text("Detections: \(detections.count)")

            Button("Select Image") {
                // Image picker logic
            }
        }
        .onChange(of: image) { newImage in
            guard let newImage = newImage else { return }

            Task {
                do {
                    detections = try model.predict(image: newImage, confThreshold: 0.25)
                    annotatedImage = model.visualize(results: detections, image: newImage)
                } catch {
                    print("Error: \(error)")
                }
            }
        }
    }
}
```

## Performance Tips

1. **Use Neural Engine**: CoreML automatically uses Neural Engine on A12+ devices
2. **Image size**: Use 640x640 for best speed/accuracy tradeoff
3. **Batch processing**: Process multiple images in parallel with `async/await`
4. **Cache model**: Initialize once, reuse for all predictions
5. **Lower threshold**: Higher `confThreshold` = faster postprocessing

## Troubleshooting

**Q: "Model failed to load"**
- Ensure `.mlmodel` or `.mlpackage` is added to Xcode target
- Check that `metadata.json` exists in the same directory
- Verify model is compatible with iOS/macOS version

**Q: "Low FPS on camera"**
- Use lower resolution preset (e.g., `.vga640x480`)
- Increase `confThreshold` to reduce postprocessing time
- Use `.background` quality of service for inference queue

**Q: "Model output mismatch"**
- Verify input shape in metadata.json matches model
- Check that preprocessing parameters are correct
- Ensure model was exported with `format='coreml'` from Ultralytics

**Q: "Memory issues"**
- Release images after processing
- Use `autoreleasepool` in loops
- Lower camera resolution

## Example Project

See example iOS app in `Example/` directory.

## License

See parent LICENSE file.
