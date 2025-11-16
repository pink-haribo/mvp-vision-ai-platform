# Kotlin TensorFlow Lite Wrapper

Easy-to-use Kotlin wrapper for running YOLO models on Android using TensorFlow Lite.

## Requirements

- Android SDK 21+ (Android 5.0 Lollipop)
- Kotlin 1.8+
- TensorFlow Lite 2.13+
- Android Studio Flamingo or later

## Installation

### Gradle

Add to your `build.gradle`:

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'  // For GPU acceleration
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
}
```

### Manual Integration

1. Copy `ModelWrapper.kt` to your Android project
2. Copy your `.tflite` model to `assets/` directory
3. Copy `metadata.json` to `assets/` directory

## Quick Start

```kotlin
import com.visionai.yolo.YOLOInference

class MainActivity : AppCompatActivity() {
    private lateinit var model: YOLOInference

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize model
        model = YOLOInference(
            context = this,
            modelPath = "model.tflite",
            metadataPath = "metadata.json",
            useGPU = true
        )

        // Load image
        val bitmap = BitmapFactory.decodeResource(resources, R.drawable.test_image)

        // Run inference
        val detections = model.predict(
            bitmap = bitmap,
            confThreshold = 0.25f,
            iouThreshold = 0.45f
        )

        Log.d("YOLO", "Detections: ${detections.size}")

        // Visualize
        val annotated = model.visualize(detections, bitmap)
        imageView.setImageBitmap(annotated)
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }
}
```

## API Reference

### YOLOInference

```kotlin
val model = YOLOInference(
    context: Context,          // Application context
    modelPath: String,         // Path to .tflite model in assets
    metadataPath: String? = null,  // Path to metadata.json (optional)
    useGPU: Boolean = true     // Enable GPU acceleration
)
```

### predict()

```kotlin
val detections = model.predict(
    bitmap: Bitmap,
    confThreshold: Float = 0.25f,
    iouThreshold: Float = 0.45f,
    maxDet: Int = 300
)
```

**Detection structure:**
```kotlin
data class Detection(
    val box: RectF,        // Bounding box
    val confidence: Float, // Confidence score
    val classId: Int,     // Class ID
    val className: String? // Class name (from metadata)
)
```

### visualize()

```kotlin
val annotated = model.visualize(
    results: List<Detection>,
    bitmap: Bitmap,
    showLabels: Boolean = true,
    showConf: Boolean = true
)
```

## Advanced Usage

### Camera Integration

```kotlin
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider

class CameraActivity : AppCompatActivity() {
    private lateinit var model: YOLOInference

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        model = YOLOInference(this, "model.tflite", useGPU = true)

        startCamera()
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val preview = Preview.Builder().build()
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(android.util.Size(640, 640))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
                processImage(imageProxy)
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageAnalysis
                )
            } catch (e: Exception) {
                Log.e("Camera", "Binding failed", e)
            }

        }, ContextCompat.getMainExecutor(this))
    }

    @androidx.camera.core.ExperimentalGetImage
    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image
        if (mediaImage != null) {
            val bitmap = mediaImage.toBitmap()

            val detections = model.predict(bitmap, confThreshold = 0.3f)

            runOnUiThread {
                val annotated = model.visualize(detections, bitmap)
                imageView.setImageBitmap(annotated)
            }
        }

        imageProxy.close()
    }
}

// Extension function to convert Image to Bitmap
fun Image.toBitmap(): Bitmap {
    val yBuffer = planes[0].buffer
    val uBuffer = planes[1].buffer
    val vBuffer = planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
    val out = ByteArrayOutputStream()
    yuvImage.compressToJpeg(Rect(0, 0, width, height), 100, out)
    val imageBytes = out.toByteArray()

    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}
```

### Coroutines and Flow

```kotlin
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.*

class DetectionViewModel : ViewModel() {
    private val model by lazy {
        YOLOInference(context, "model.tflite")
    }

    fun detectObjects(bitmap: Bitmap): Flow<List<Detection>> = flow {
        val detections = withContext(Dispatchers.Default) {
            model.predict(bitmap, confThreshold = 0.25f)
        }
        emit(detections)
    }.flowOn(Dispatchers.IO)

    override fun onCleared() {
        super.onCleared()
        model.close()
    }
}

// Usage in Activity/Fragment
lifecycleScope.launch {
    viewModel.detectObjects(bitmap).collect { detections ->
        Log.d("Detections", "Found: ${detections.size}")
        updateUI(detections)
    }
}
```

### Batch Processing

```kotlin
suspend fun processBatch(images: List<Bitmap>) = coroutineScope {
    images.map { bitmap ->
        async(Dispatchers.Default) {
            model.predict(bitmap)
        }
    }.awaitAll()
}

// Usage
lifecycleScope.launch {
    val results = processBatch(imageList)
    results.forEachIndexed { index, detections ->
        Log.d("Batch", "Image $index: ${detections.size} detections")
    }
}
```

### Custom Postprocessing

```kotlin
// Filter by class
val personDetections = detections.filter { it.classId == 0 }

// Filter by confidence
val highConfDetections = detections.filter { it.confidence > 0.5f }

// Filter by area
val largeDetections = detections.filter {
    val area = it.box.width() * it.box.height()
    area > 10000f
}

// Custom visualization
fun customVisualize(detections: List<Detection>, bitmap: Bitmap): Bitmap {
    val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
    val canvas = Canvas(output)

    val paint = Paint().apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        color = Color.GREEN
    }

    for (detection in detections.filter { it.classId == 0 }) {
        canvas.drawRect(detection.box, paint)

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 48f
            typeface = Typeface.DEFAULT_BOLD
        }

        val label = "Person ${String.format("%.2f", detection.confidence)}"
        canvas.drawText(label, detection.box.left, detection.box.top - 10, textPaint)
    }

    return output
}
```

### Jetpack Compose Integration

```kotlin
import androidx.compose.runtime.*
import androidx.compose.ui.graphics.asImageBitmap

@Composable
fun DetectionScreen(model: YOLOInference) {
    var bitmap by remember { mutableStateOf<Bitmap?>(null) }
    var annotatedBitmap by remember { mutableStateOf<Bitmap?>(null) }
    var detectionCount by remember { mutableStateOf(0) }

    LaunchedEffect(bitmap) {
        bitmap?.let {
            val detections = withContext(Dispatchers.Default) {
                model.predict(it, confThreshold = 0.25f)
            }
            detectionCount = detections.size
            annotatedBitmap = model.visualize(detections, it)
        }
    }

    Column {
        annotatedBitmap?.let {
            Image(
                bitmap = it.asImageBitmap(),
                contentDescription = "Detected objects",
                modifier = Modifier.fillMaxWidth()
            )
        }

        Text("Detections: $detectionCount")

        Button(onClick = { /* Image picker logic */ }) {
            Text("Select Image")
        }
    }
}
```

## Performance Optimization

1. **Use GPU**: ~5-10x faster on devices with GPU delegate support
   ```kotlin
   val model = YOLOInference(context, "model.tflite", useGPU = true)
   ```

2. **NNAPI Delegate**: For devices with Neural Processing Units
   ```kotlin
   // Add to Interpreter.Options
   options.setUseNNAPI(true)
   ```

3. **Quantized Models**: Use INT8 quantization for smaller size and faster inference
   ```python
   # Export with quantization
   model.export(format='tflite', int8=True)
   ```

4. **Image Resolution**: Balance speed vs accuracy
   ```kotlin
   // Lower resolution = faster
   val resized = Bitmap.createScaledBitmap(bitmap, 320, 320, true)
   val detections = model.predict(resized)
   ```

5. **Multithreading**: Process multiple images in parallel
   ```kotlin
   options.setNumThreads(4)
   ```

## Troubleshooting

**Q: "Model failed to load"**
- Ensure `.tflite` file is in `assets/` directory
- Check that `metadata.json` exists in `assets/`
- Verify model is compatible with TFLite version

**Q: "GPU delegate not available"**
- Some devices don't support GPU delegate
- Fall back to CPU: `YOLOInference(context, "model.tflite", useGPU = false)`

**Q: "Low FPS on camera"**
- Use lower resolution: `setTargetResolution(Size(320, 320))`
- Increase `confThreshold` to reduce postprocessing
- Use quantized model (INT8)

**Q: "Out of memory"**
- Release bitmaps after use: `bitmap.recycle()`
- Use smaller image resolution
- Close model when done: `model.close()`

**Q: "Incorrect detections"**
- Verify input preprocessing matches training
- Check metadata.json has correct parameters
- Ensure model was exported with `format='tflite'`

## Example Project

See example Android app in `Example/` directory.

## Dependencies

```gradle
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.13.0'
    implementation 'org.tensorflow:tensorflow-lite-support:0.4.4'
    implementation 'androidx.camera:camera-core:1.3.0'
    implementation 'androidx.camera:camera-camera2:1.3.0'
    implementation 'androidx.camera:camera-lifecycle:1.3.0'
    implementation 'androidx.camera:camera-view:1.3.0'
    implementation 'org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3'
}
```

## License

See parent LICENSE file.
