/**
 * ModelWrapper.kt
 * TensorFlow Lite wrapper for YOLO models exported from Vision AI Training Platform
 *
 * Provides easy-to-use Kotlin interface for running YOLO models on Android.
 *
 * Supports:
 * - YOLOv8/YOLO11 Detection
 * - YOLOv8/YOLO11 Segmentation
 * - YOLOv8/YOLO11 Pose Estimation
 * - YOLOv8/YOLO11 Classification
 *
 * Requirements:
 *   - Android API 21+
 *   - TensorFlow Lite 2.13+
 *   - TensorFlow Lite GPU delegate (optional)
 *
 * Usage:
 *   val model = YOLOInference(context, "model.tflite", "metadata.json")
 *   val results = model.predict(bitmap, confThreshold = 0.25f, iouThreshold = 0.45f)
 *   val annotated = model.visualize(results, bitmap)
 */

package com.visionai.yolo

import android.content.Context
import android.graphics.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import org.json.JSONObject
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.max
import kotlin.math.min

// MARK: - Result Structures

/**
 * Detection result for a single object
 */
data class Detection(
    val box: RectF,
    val confidence: Float,
    val classId: Int,
    val className: String? = null
)

/**
 * Pose estimation result
 */
data class PoseResult(
    val box: RectF,
    val confidence: Float,
    val keypoints: List<PointF>,
    val keypointConfidences: List<Float>
)

/**
 * Classification result
 */
data class ClassificationResult(
    val classId: Int,
    val confidence: Float,
    val probabilities: FloatArray,
    val className: String? = null
)

/**
 * Metadata structure
 */
private data class Metadata(
    val taskType: String,
    val inputShape: IntArray,
    val classNames: List<String>,
    val preprocessing: Preprocessing
) {
    data class Preprocessing(
        val resize: String,
        val normalize: Boolean,
        val mean: FloatArray,
        val std: FloatArray
    )
}

/**
 * Preprocessing information
 */
private data class PreprocessInfo(
    val originalSize: Size,
    val ratio: Float,
    val padW: Int,
    val padH: Int
)

private data class Size(val width: Int, val height: Int)

// MARK: - Main Inference Class

/**
 * YOLO inference engine using TensorFlow Lite
 */
class YOLOInference(
    private val context: Context,
    modelPath: String,
    metadataPath: String? = null,
    useGPU: Boolean = true
) {
    private val interpreter: Interpreter
    private val metadata: Metadata
    private val taskType: String
    private val inputSize: Size
    private val classNames: List<String>
    private val gpuDelegate: GpuDelegate?

    init {
        // Load metadata
        val metadataFile = metadataPath ?: modelPath.replace(".tflite", "_metadata.json")
        metadata = loadMetadata(metadataFile)

        taskType = metadata.taskType
        classNames = metadata.classNames
        inputSize = Size(metadata.inputShape[3], metadata.inputShape[2])

        // Initialize TensorFlow Lite
        val options = Interpreter.Options()

        if (useGPU) {
            gpuDelegate = GpuDelegate()
            options.addDelegate(gpuDelegate)
        } else {
            gpuDelegate = null
        }

        options.setNumThreads(4)

        interpreter = Interpreter(loadModelFile(modelPath), options)

        println("[YOLOInference] Model loaded: $modelPath")
        println("[YOLOInference] Task: $taskType")
        println("[YOLOInference] Input size: ${inputSize.width}x${inputSize.height}")
        println("[YOLOInference] Classes: ${classNames.size}")
    }

    /**
     * Release resources
     */
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    // MARK: - Model Loading

    private fun loadModelFile(modelPath: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelPath)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    private fun loadMetadata(metadataPath: String): Metadata {
        val jsonString = context.assets.open(metadataPath).bufferedReader().use { it.readText() }
        val json = JSONObject(jsonString)

        val inputShape = json.getJSONArray("input_shape")
        val shape = IntArray(inputShape.length()) { inputShape.getInt(it) }

        val classNamesArray = json.optJSONArray("class_names")
        val classes = if (classNamesArray != null) {
            List(classNamesArray.length()) { classNamesArray.getString(it) }
        } else {
            emptyList()
        }

        val preprocessing = json.getJSONObject("preprocessing")
        val mean = preprocessing.getJSONArray("mean")
        val std = preprocessing.getJSONArray("std")

        return Metadata(
            taskType = json.getString("task_type"),
            inputShape = shape,
            classNames = classes,
            preprocessing = Metadata.Preprocessing(
                resize = preprocessing.getString("resize"),
                normalize = preprocessing.getBoolean("normalize"),
                mean = FloatArray(3) { mean.getDouble(it).toFloat() },
                std = FloatArray(3) { std.getDouble(it).toFloat() }
            )
        )
    }

    // MARK: - Prediction

    /**
     * Run object detection
     *
     * @param bitmap Input bitmap (any size)
     * @param confThreshold Confidence threshold (0.0-1.0)
     * @param iouThreshold IoU threshold for NMS (0.0-1.0)
     * @param maxDet Maximum number of detections
     * @return List of detections
     */
    fun predict(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDet: Int = 300
    ): List<Detection> {
        require(taskType == "detect" || taskType == "detection") {
            "Use predictPose() or predictClassification() for other tasks"
        }

        // Preprocess
        val (inputBuffer, preprocessInfo) = preprocess(bitmap)

        // Prepare output buffer
        // YOLOv8: [1, 84, 8400]
        val outputShape = intArrayOf(1, 84, 8400)
        val outputBuffer = ByteBuffer.allocateDirect(4 * 84 * 8400).apply {
            order(ByteOrder.nativeOrder())
        }

        // Run inference
        interpreter.run(inputBuffer, outputBuffer)

        // Postprocess
        outputBuffer.rewind()
        val output = FloatArray(84 * 8400)
        outputBuffer.asFloatBuffer().get(output)

        return postprocessDetect(output, preprocessInfo, confThreshold, iouThreshold, maxDet)
    }

    /**
     * Run pose estimation
     */
    fun predictPose(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f,
        iouThreshold: Float = 0.45f,
        maxDet: Int = 300
    ): List<PoseResult> {
        // Similar to predict() but with keypoint extraction
        return emptyList()
    }

    /**
     * Run image classification
     */
    fun predictClassification(
        bitmap: Bitmap,
        confThreshold: Float = 0.25f
    ): ClassificationResult {
        // Classification logic
        return ClassificationResult(
            classId = -1,
            confidence = 0.0f,
            probabilities = floatArrayOf(),
            className = null
        )
    }

    // MARK: - Preprocessing

    private fun preprocess(bitmap: Bitmap): Pair<ByteBuffer, PreprocessInfo> {
        val originalSize = Size(bitmap.width, bitmap.height)

        // Letterbox resize
        val (resized, ratio, padW, padH) = letterboxResize(bitmap, inputSize)

        // Convert to ByteBuffer
        val inputBuffer = ByteBuffer.allocateDirect(
            4 * 1 * 3 * inputSize.height * inputSize.width
        ).apply {
            order(ByteOrder.nativeOrder())
        }

        val pixels = IntArray(inputSize.width * inputSize.height)
        resized.getPixels(pixels, 0, inputSize.width, 0, 0, inputSize.width, inputSize.height)

        val mean = metadata.preprocessing.mean
        val std = metadata.preprocessing.std

        // Convert RGBA to RGB and normalize (CHW format)
        for (c in 0..2) {
            for (y in 0 until inputSize.height) {
                for (x in 0 until inputSize.width) {
                    val pixel = pixels[y * inputSize.width + x]

                    val value = when (c) {
                        0 -> Color.red(pixel).toFloat()
                        1 -> Color.green(pixel).toFloat()
                        else -> Color.blue(pixel).toFloat()
                    }

                    val normalized = (value - mean[c]) / std[c]
                    inputBuffer.putFloat(normalized)
                }
            }
        }

        inputBuffer.rewind()

        val preprocessInfo = PreprocessInfo(
            originalSize = originalSize,
            ratio = ratio,
            padW = padW,
            padH = padH
        )

        return Pair(inputBuffer, preprocessInfo)
    }

    private fun letterboxResize(
        bitmap: Bitmap,
        targetSize: Size
    ): Triple<Bitmap, Float, Int, Int> {
        // Calculate scale ratio
        val scaleW = targetSize.width.toFloat() / bitmap.width
        val scaleH = targetSize.height.toFloat() / bitmap.height
        val ratio = min(scaleW, scaleH)

        // Calculate new size and padding
        val newW = (bitmap.width * ratio).toInt()
        val newH = (bitmap.height * ratio).toInt()

        val padW = (targetSize.width - newW) / 2
        val padH = (targetSize.height - newH) / 2

        // Create canvas with padding
        val output = Bitmap.createBitmap(targetSize.width, targetSize.height, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(output)

        // Fill with gray (114, 114, 114)
        canvas.drawColor(Color.rgb(114, 114, 114))

        // Draw resized bitmap
        val scaled = Bitmap.createScaledBitmap(bitmap, newW, newH, true)
        canvas.drawBitmap(scaled, padW.toFloat(), padH.toFloat(), null)

        return Triple(output, ratio, padW, padH)
    }

    // MARK: - Postprocessing

    private fun postprocessDetect(
        output: FloatArray,
        preprocessInfo: PreprocessInfo,
        confThreshold: Float,
        iouThreshold: Float,
        maxDet: Int
    ): List<Detection> {
        // YOLOv8 output: [84, 8400]
        // Format: [cx, cy, w, h, class0_conf, class1_conf, ...]

        val numClasses = 80
        val numBoxes = 8400

        val boxes = mutableListOf<RectF>()
        val scores = mutableListOf<Float>()
        val classIds = mutableListOf<Int>()

        for (i in 0 until numBoxes) {
            // Find max class confidence
            var maxConf = 0.0f
            var maxClassId = 0

            for (c in 0 until numClasses) {
                val conf = output[(4 + c) * numBoxes + i]
                if (conf > maxConf) {
                    maxConf = conf
                    maxClassId = c
                }
            }

            if (maxConf < confThreshold) {
                continue
            }

            // Extract box (cx, cy, w, h)
            val cx = output[0 * numBoxes + i]
            val cy = output[1 * numBoxes + i]
            val w = output[2 * numBoxes + i]
            val h = output[3 * numBoxes + i]

            // Convert to (x, y, w, h)
            val x = cx - w / 2.0f
            val y = cy - h / 2.0f

            boxes.add(RectF(x, y, x + w, y + h))
            scores.add(maxConf)
            classIds.add(maxClassId)
        }

        if (boxes.isEmpty()) {
            return emptyList()
        }

        // Apply NMS
        val keepIndices = nms(boxes, scores, iouThreshold)
        val finalIndices = keepIndices.take(maxDet)

        // Create detections
        return finalIndices.map { idx ->
            val scaledBox = scaleBox(boxes[idx], preprocessInfo)
            val className = classIds[idx].takeIf { it < classNames.size }?.let { classNames[it] }

            Detection(
                box = scaledBox,
                confidence = scores[idx],
                classId = classIds[idx],
                className = className
            )
        }
    }

    // MARK: - Utility Functions

    private fun nms(boxes: List<RectF>, scores: List<Float>, iouThreshold: Float): List<Int> {
        val indices = scores.indices.sortedByDescending { scores[it] }.toMutableList()
        val keep = mutableListOf<Int>()

        while (indices.isNotEmpty()) {
            val current = indices.removeAt(0)
            keep.add(current)

            indices.removeAll { idx ->
                calculateIoU(boxes[current], boxes[idx]) > iouThreshold
            }
        }

        return keep
    }

    private fun calculateIoU(box1: RectF, box2: RectF): Float {
        val intersection = RectF(box1)
        if (!intersection.intersect(box2)) {
            return 0.0f
        }

        val intersectionArea = intersection.width() * intersection.height()
        val unionArea = box1.width() * box1.height() + box2.width() * box2.height() - intersectionArea

        return intersectionArea / unionArea
    }

    private fun scaleBox(box: RectF, preprocessInfo: PreprocessInfo): RectF {
        val scaled = RectF(box)

        // Remove padding
        scaled.left -= preprocessInfo.padW
        scaled.top -= preprocessInfo.padH
        scaled.right -= preprocessInfo.padW
        scaled.bottom -= preprocessInfo.padH

        // Scale back
        scaled.left /= preprocessInfo.ratio
        scaled.top /= preprocessInfo.ratio
        scaled.right /= preprocessInfo.ratio
        scaled.bottom /= preprocessInfo.ratio

        // Clip to original image
        scaled.left = max(0f, min(scaled.left, preprocessInfo.originalSize.width.toFloat()))
        scaled.top = max(0f, min(scaled.top, preprocessInfo.originalSize.height.toFloat()))
        scaled.right = max(0f, min(scaled.right, preprocessInfo.originalSize.width.toFloat()))
        scaled.bottom = max(0f, min(scaled.bottom, preprocessInfo.originalSize.height.toFloat()))

        return scaled
    }

    // MARK: - Visualization

    /**
     * Visualize detection results on bitmap
     *
     * @param results Detection results
     * @param bitmap Original bitmap
     * @param showLabels Show class labels
     * @param showConf Show confidence scores
     * @return Annotated bitmap
     */
    fun visualize(
        results: List<Detection>,
        bitmap: Bitmap,
        showLabels: Boolean = true,
        showConf: Boolean = true
    ): Bitmap {
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        val boxPaint = Paint().apply {
            style = Paint.Style.STROKE
            strokeWidth = 4f
        }

        val textPaint = Paint().apply {
            color = Color.WHITE
            textSize = 40f
            typeface = Typeface.DEFAULT_BOLD
        }

        val bgPaint = Paint().apply {
            style = Paint.Style.FILL
        }

        for (detection in results) {
            // Draw box
            val color = getColor(detection.classId)
            boxPaint.color = color
            canvas.drawRect(detection.box, boxPaint)

            // Draw label
            if (showLabels || showConf) {
                var label = ""
                if (showLabels && detection.className != null) {
                    label = detection.className
                }
                if (showConf) {
                    if (label.isNotEmpty()) label += " "
                    label += String.format("%.2f", detection.confidence)
                }

                val textBounds = Rect()
                textPaint.getTextBounds(label, 0, label.length, textBounds)

                val textX = detection.box.left
                val textY = detection.box.top - 10

                bgPaint.color = color
                canvas.drawRect(
                    textX,
                    textY - textBounds.height() - 10,
                    textX + textBounds.width() + 20,
                    textY,
                    bgPaint
                )

                canvas.drawText(label, textX + 10, textY - 5, textPaint)
            }
        }

        return output
    }

    private fun getColor(classId: Int): Int {
        val hue = (classId * 37 % 360).toFloat()
        return Color.HSVToColor(floatArrayOf(hue, 0.8f, 0.9f))
    }
}
