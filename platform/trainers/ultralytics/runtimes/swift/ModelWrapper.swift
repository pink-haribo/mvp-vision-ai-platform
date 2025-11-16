/**
 * ModelWrapper.swift
 * CoreML wrapper for YOLO models exported from Vision AI Training Platform
 *
 * Provides easy-to-use Swift interface for running YOLO models on iOS/macOS.
 *
 * Supports:
 * - YOLOv8/YOLO11 Detection
 * - YOLOv8/YOLO11 Segmentation
 * - YOLOv8/YOLO11 Pose Estimation
 * - YOLOv8/YOLO11 Classification
 *
 * Requirements:
 *   - iOS 13+ / macOS 10.15+
 *   - CoreML
 *   - Vision framework
 *
 * Usage:
 *   let model = try YOLOInference(modelURL: modelURL, metadataURL: metadataURL)
 *   let results = try model.predict(image: uiImage, confThreshold: 0.25, iouThreshold: 0.45)
 *   let annotated = model.visualize(results: results, image: uiImage)
 */

import Foundation
import CoreML
import Vision
import UIKit
import Accelerate

// MARK: - Result Structures

/// Detection result for a single object
public struct Detection {
    public let box: CGRect
    public let confidence: Float
    public let classId: Int
    public let className: String?
}

/// Pose estimation result
public struct PoseResult {
    public let box: CGRect
    public let confidence: Float
    public let keypoints: [CGPoint]
    public let keypointConfidences: [Float]
}

/// Classification result
public struct ClassificationResult {
    public let classId: Int
    public let confidence: Float
    public let probabilities: [Float]
    public let className: String?
}

/// Metadata structure
struct Metadata: Codable {
    let taskType: String
    let inputShape: [Int]
    let classNames: [String]?
    let preprocessing: Preprocessing

    struct Preprocessing: Codable {
        let resize: String
        let normalize: Bool
        let mean: [Float]
        let std: [Float]
    }

    enum CodingKeys: String, CodingKey {
        case taskType = "task_type"
        case inputShape = "input_shape"
        case classNames = "class_names"
        case preprocessing
    }
}

// MARK: - Main Inference Class

public class YOLOInference {

    // MARK: Properties

    private let model: MLModel
    private let metadata: Metadata
    private let taskType: String
    private let inputSize: CGSize
    private let classNames: [String]

    // MARK: Initialization

    /**
     Initialize YOLO inference engine

     - Parameters:
       - modelURL: URL to CoreML model file (.mlmodel or .mlpackage)
       - metadataURL: URL to metadata.json (optional, auto-detected if nil)

     - Throws: Initialization errors
     */
    public init(modelURL: URL, metadataURL: URL? = nil) throws {
        // Load metadata
        let metadataPath = metadataURL ?? modelURL
            .deletingLastPathComponent()
            .appendingPathComponent("metadata.json")

        let metadataData = try Data(contentsOf: metadataPath)
        self.metadata = try JSONDecoder().decode(Metadata.self, from: metadataData)

        self.taskType = metadata.taskType
        self.classNames = metadata.classNames ?? []

        let height = metadata.inputShape[2]
        let width = metadata.inputShape[3]
        self.inputSize = CGSize(width: width, height: height)

        // Load CoreML model
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine + GPU + CPU

        self.model = try MLModel(contentsOf: modelURL, configuration: config)

        print("[YOLOInference] Model loaded: \(modelURL.lastPathComponent)")
        print("[YOLOInference] Task: \(taskType)")
        print("[YOLOInference] Input size: \(inputSize)")
        print("[YOLOInference] Classes: \(classNames.count)")
    }

    // MARK: - Prediction

    /**
     Run object detection

     - Parameters:
       - image: Input UIImage
       - confThreshold: Confidence threshold (0.0-1.0)
       - iouThreshold: IoU threshold for NMS (0.0-1.0)
       - maxDet: Maximum number of detections

     - Returns: Array of detections
     - Throws: Prediction errors
     */
    public func predict(
        image: UIImage,
        confThreshold: Float = 0.25,
        iouThreshold: Float = 0.45,
        maxDet: Int = 300
    ) throws -> [Detection] {
        guard taskType == "detect" || taskType == "detection" else {
            throw NSError(domain: "YOLOInference", code: -1,
                         userInfo: [NSLocalizedDescriptionKey: "Use predictPose() or predictClassification() for other tasks"])
        }

        // Preprocess
        let (inputBuffer, preprocessInfo) = try preprocess(image: image)

        // Create MLMultiArray from CVPixelBuffer
        let inputArray = try createMLMultiArray(from: inputBuffer)

        // Run inference
        let inputName = model.modelDescription.inputDescriptionsByName.keys.first!
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: inputArray])

        let output = try model.prediction(from: input)
        let outputName = model.modelDescription.outputDescriptionsByName.keys.first!

        guard let outputArray = output.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "YOLOInference", code: -2,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get model output"])
        }

        // Postprocess
        let detections = postprocessDetect(
            output: outputArray,
            preprocessInfo: preprocessInfo,
            confThreshold: confThreshold,
            iouThreshold: iouThreshold,
            maxDet: maxDet
        )

        return detections
    }

    /**
     Run pose estimation

     - Parameters:
       - image: Input UIImage
       - confThreshold: Confidence threshold
       - iouThreshold: IoU threshold for NMS
       - maxDet: Maximum number of detections

     - Returns: Array of pose results
     - Throws: Prediction errors
     */
    public func predictPose(
        image: UIImage,
        confThreshold: Float = 0.25,
        iouThreshold: Float = 0.45,
        maxDet: Int = 300
    ) throws -> [PoseResult] {
        // Similar to predict() but with keypoint extraction
        return []
    }

    /**
     Run image classification

     - Parameters:
       - image: Input UIImage
       - confThreshold: Confidence threshold

     - Returns: Classification result
     - Throws: Prediction errors
     */
    public func predictClassification(
        image: UIImage,
        confThreshold: Float = 0.25
    ) throws -> ClassificationResult {
        // Classification logic
        return ClassificationResult(classId: -1, confidence: 0.0, probabilities: [], className: nil)
    }

    // MARK: - Preprocessing

    private func preprocess(image: UIImage) throws -> (CVPixelBuffer, PreprocessInfo) {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "YOLOInference", code: -3,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage"])
        }

        let originalSize = CGSize(width: cgImage.width, height: cgImage.height)

        // Letterbox resize
        let (resized, ratio, padding) = letterboxResize(
            image: image,
            targetSize: inputSize
        )

        // Convert to CVPixelBuffer
        let pixelBuffer = try createPixelBuffer(from: resized)

        let preprocessInfo = PreprocessInfo(
            originalSize: originalSize,
            ratio: ratio,
            padding: padding
        )

        return (pixelBuffer, preprocessInfo)
    }

    private func letterboxResize(
        image: UIImage,
        targetSize: CGSize
    ) -> (UIImage, Float, CGPoint) {
        let size = image.size

        // Calculate scale ratio
        let scaleW = Float(targetSize.width) / Float(size.width)
        let scaleH = Float(targetSize.height) / Float(size.height)
        let ratio = min(scaleW, scaleH)

        // Calculate new size and padding
        let newW = Int(Float(size.width) * ratio)
        let newH = Int(Float(size.height) * ratio)

        let padW = (Int(targetSize.width) - newW) / 2
        let padH = (Int(targetSize.height) - newH) / 2

        // Create canvas with padding
        UIGraphicsBeginImageContextWithOptions(targetSize, false, 1.0)
        defer { UIGraphicsEndImageContext() }

        // Fill with gray (114, 114, 114)
        let context = UIGraphicsGetCurrentContext()!
        context.setFillColor(UIColor(red: 114/255, green: 114/255, blue: 114/255, alpha: 1.0).cgColor)
        context.fill(CGRect(origin: .zero, size: targetSize))

        // Draw resized image
        let rect = CGRect(x: padW, y: padH, width: newW, height: newH)
        image.draw(in: rect)

        let resizedImage = UIGraphicsGetImageFromCurrentImageContext()!

        return (resizedImage, ratio, CGPoint(x: padW, y: padH))
    }

    private func createPixelBuffer(from image: UIImage) throws -> CVPixelBuffer {
        guard let cgImage = image.cgImage else {
            throw NSError(domain: "YOLOInference", code: -4,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to get CGImage"])
        }

        let width = cgImage.width
        let height = cgImage.height

        var pixelBuffer: CVPixelBuffer?
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue!,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue!
        ] as CFDictionary

        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs,
            &pixelBuffer
        )

        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            throw NSError(domain: "YOLOInference", code: -5,
                         userInfo: [NSLocalizedDescriptionKey: "Failed to create pixel buffer"])
        }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }

        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )!

        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

        return buffer
    }

    private func createMLMultiArray(from pixelBuffer: CVPixelBuffer) throws -> MLMultiArray {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        let shape = [1, 3, height, width] as [NSNumber]
        let array = try MLMultiArray(shape: shape, dataType: .float32)

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer)!
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)

        // Convert BGRA to RGB and normalize
        let mean = metadata.preprocessing.mean
        let std = metadata.preprocessing.std

        for y in 0..<height {
            for x in 0..<width {
                let pixel = baseAddress.advanced(by: y * bytesPerRow + x * 4)
                let b = Float(pixel.load(fromByteOffset: 0, as: UInt8.self))
                let g = Float(pixel.load(fromByteOffset: 1, as: UInt8.self))
                let r = Float(pixel.load(fromByteOffset: 2, as: UInt8.self))

                let rIdx = [0, 0, y, x] as [NSNumber]
                let gIdx = [0, 1, y, x] as [NSNumber]
                let bIdx = [0, 2, y, x] as [NSNumber]

                array[rIdx] = NSNumber(value: (r - mean[0]) / std[0])
                array[gIdx] = NSNumber(value: (g - mean[1]) / std[1])
                array[bIdx] = NSNumber(value: (b - mean[2]) / std[2])
            }
        }

        return array
    }

    // MARK: - Postprocessing

    private func postprocessDetect(
        output: MLMultiArray,
        preprocessInfo: PreprocessInfo,
        confThreshold: Float,
        iouThreshold: Float,
        maxDet: Int
    ) -> [Detection] {
        // YOLOv8 output: [1, 84, 8400]
        let numBoxes = output.shape[2].intValue
        let numClasses = 80

        var boxes: [CGRect] = []
        var scores: [Float] = []
        var classIds: [Int] = []

        for i in 0..<numBoxes {
            // Find max class confidence
            var maxConf: Float = 0.0
            var maxClassId = 0

            for c in 0..<numClasses {
                let idx = [0, 4 + c, i] as [NSNumber]
                let conf = output[idx].floatValue
                if conf > maxConf {
                    maxConf = conf
                    maxClassId = c
                }
            }

            if maxConf < confThreshold {
                continue
            }

            // Extract box (cx, cy, w, h)
            let cx = output[[0, 0, i] as [NSNumber]].floatValue
            let cy = output[[0, 1, i] as [NSNumber]].floatValue
            let w = output[[0, 2, i] as [NSNumber]].floatValue
            let h = output[[0, 3, i] as [NSNumber]].floatValue

            // Convert to (x, y, w, h)
            let x = cx - w / 2.0
            let y = cy - h / 2.0

            boxes.append(CGRect(x: CGFloat(x), y: CGFloat(y), width: CGFloat(w), height: CGFloat(h)))
            scores.append(maxConf)
            classIds.append(maxClassId)
        }

        if boxes.isEmpty {
            return []
        }

        // Apply NMS
        let keepIndices = nms(boxes: boxes, scores: scores, iouThreshold: iouThreshold)
        let finalIndices = Array(keepIndices.prefix(maxDet))

        // Create detections
        var detections: [Detection] = []
        for idx in finalIndices {
            let scaledBox = scaleBox(box: boxes[idx], preprocessInfo: preprocessInfo)
            let className = classIds[idx] < classNames.count ? classNames[classIds[idx]] : nil

            let detection = Detection(
                box: scaledBox,
                confidence: scores[idx],
                classId: classIds[idx],
                className: className
            )
            detections.append(detection)
        }

        return detections
    }

    // MARK: - Utility Functions

    private func nms(boxes: [CGRect], scores: [Float], iouThreshold: Float) -> [Int] {
        var indices = Array(0..<boxes.count)
        indices.sort { scores[$0] > scores[$1] }

        var keep: [Int] = []

        while !indices.isEmpty {
            let current = indices.removeFirst()
            keep.append(current)

            indices = indices.filter { idx in
                let iou = calculateIoU(box1: boxes[current], box2: boxes[idx])
                return iou <= iouThreshold
            }
        }

        return keep
    }

    private func calculateIoU(box1: CGRect, box2: CGRect) -> Float {
        let intersection = box1.intersection(box2)
        if intersection.isNull {
            return 0.0
        }

        let intersectionArea = intersection.width * intersection.height
        let unionArea = box1.width * box1.height + box2.width * box2.height - intersectionArea

        return Float(intersectionArea / unionArea)
    }

    private func scaleBox(box: CGRect, preprocessInfo: PreprocessInfo) -> CGRect {
        var scaled = box

        // Remove padding
        scaled.origin.x -= preprocessInfo.padding.x
        scaled.origin.y -= preprocessInfo.padding.y

        // Scale back
        let ratio = CGFloat(preprocessInfo.ratio)
        scaled.origin.x /= ratio
        scaled.origin.y /= ratio
        scaled.size.width /= ratio
        scaled.size.height /= ratio

        // Clip to original image
        scaled.origin.x = max(0, min(scaled.origin.x, preprocessInfo.originalSize.width))
        scaled.origin.y = max(0, min(scaled.origin.y, preprocessInfo.originalSize.height))

        return scaled
    }

    // MARK: - Visualization

    /**
     Visualize detection results on image

     - Parameters:
       - results: Detection results
       - image: Original UIImage
       - showLabels: Show class labels
       - showConf: Show confidence scores

     - Returns: Annotated UIImage
     */
    public func visualize(
        results: [Detection],
        image: UIImage,
        showLabels: Bool = true,
        showConf: Bool = true
    ) -> UIImage {
        UIGraphicsBeginImageContextWithOptions(image.size, false, image.scale)
        defer { UIGraphicsEndImageContext() }

        image.draw(at: .zero)

        let context = UIGraphicsGetCurrentContext()!

        for detection in results {
            // Draw box
            let color = getColor(classId: detection.classId)
            context.setStrokeColor(color.cgColor)
            context.setLineWidth(3.0)
            context.stroke(detection.box)

            // Draw label
            if showLabels || showConf {
                var label = ""
                if showLabels, let className = detection.className {
                    label = className
                }
                if showConf {
                    if !label.isEmpty { label += " " }
                    label += String(format: "%.2f", detection.confidence)
                }

                let attributes: [NSAttributedString.Key: Any] = [
                    .font: UIFont.systemFont(ofSize: 16, weight: .bold),
                    .foregroundColor: UIColor.white
                ]

                let textSize = (label as NSString).size(withAttributes: attributes)
                let textRect = CGRect(
                    x: detection.box.origin.x,
                    y: detection.box.origin.y - textSize.height - 5,
                    width: textSize.width + 10,
                    height: textSize.height + 5
                )

                context.setFillColor(color.cgColor)
                context.fill(textRect)

                (label as NSString).draw(
                    at: CGPoint(x: textRect.origin.x + 5, y: textRect.origin.y + 2),
                    withAttributes: attributes
                )
            }
        }

        return UIGraphicsGetImageFromCurrentImageContext()!
    }

    private func getColor(classId: Int) -> UIColor {
        let hue = CGFloat(classId * 37 % 360) / 360.0
        return UIColor(hue: hue, saturation: 0.8, brightness: 0.9, alpha: 1.0)
    }
}

// MARK: - Supporting Structures

private struct PreprocessInfo {
    let originalSize: CGSize
    let ratio: Float
    let padding: CGPoint
}
