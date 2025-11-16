/**
 * @file model_wrapper.cpp
 * @brief Implementation of YOLO ONNX Runtime wrapper
 */

#include "model_wrapper.h"
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace yolo {

YOLOInference::YOLOInference(
    const std::string& modelPath,
    const std::string& metadataPath,
    bool useGPU
) {
    // Load metadata
    loadMetadata(metadataPath.empty() ?
        modelPath.substr(0, modelPath.find_last_of("/\\") + 1) + "metadata.json" :
        metadataPath
    );

    // Initialize ONNX Runtime session
    initializeSession(modelPath, useGPU);

    std::cout << "[YOLOInference] Model loaded: " << modelPath << std::endl;
    std::cout << "[YOLOInference] Task: " << taskType_ << std::endl;
    std::cout << "[YOLOInference] Input size: " << inputSize_ << std::endl;
}

YOLOInference::~YOLOInference() = default;

void YOLOInference::loadMetadata(const std::string& metadataPath) {
    std::ifstream file(metadataPath);
    if (!file.is_open()) {
        std::cerr << "[Warning] Metadata not found: " << metadataPath << std::endl;
        std::cerr << "[Warning] Using default values" << std::endl;

        // Default values
        taskType_ = "detect";
        inputSize_ = cv::Size(640, 640);
        mean_ = {0.0f, 0.0f, 0.0f};
        std_ = {255.0f, 255.0f, 255.0f};
        return;
    }

    file >> metadata_;

    // Parse metadata
    taskType_ = metadata_.value("task_type", "detect");

    auto inputShape = metadata_["input_shape"];
    inputSize_ = cv::Size(inputShape[3], inputShape[2]);

    if (metadata_.contains("class_names")) {
        classNames_ = metadata_["class_names"].get<std::vector<std::string>>();
    }

    auto preprocessing = metadata_["preprocessing"];
    mean_ = preprocessing.value("mean", std::vector<float>{0.0f, 0.0f, 0.0f});
    std_ = preprocessing.value("std", std::vector<float>{255.0f, 255.0f, 255.0f});
}

void YOLOInference::initializeSession(const std::string& modelPath, bool useGPU) {
    env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "YOLOInference");

    sessionOptions_ = std::make_unique<Ort::SessionOptions>();
    sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // Enable GPU if requested
    if (useGPU) {
        OrtCUDAProviderOptions cudaOptions;
        sessionOptions_->AppendExecutionProvider_CUDA(cudaOptions);
    }

    // Create session
#ifdef _WIN32
    std::wstring wModelPath(modelPath.begin(), modelPath.end());
    session_ = std::make_unique<Ort::Session>(*env_, wModelPath.c_str(), *sessionOptions_);
#else
    session_ = std::make_unique<Ort::Session>(*env_, modelPath.c_str(), *sessionOptions_);
#endif

    // Get input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    size_t numInputs = session_->GetInputCount();
    for (size_t i = 0; i < numInputs; i++) {
        inputNames_.push_back(session_->GetInputNameAllocated(i, allocator).get());
    }

    size_t numOutputs = session_->GetOutputCount();
    for (size_t i = 0; i < numOutputs; i++) {
        outputNames_.push_back(session_->GetOutputNameAllocated(i, allocator).get());
    }

    // Get input shape
    inputShape_ = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
}

cv::Mat YOLOInference::letterboxResize(
    const cv::Mat& image,
    cv::Size targetSize,
    PreprocessInfo& info
) {
    info.originalSize = image.size();

    // Calculate scale ratio
    float scaleW = static_cast<float>(targetSize.width) / image.cols;
    float scaleH = static_cast<float>(targetSize.height) / image.rows;
    info.ratio = std::min(scaleW, scaleH);

    // Calculate new size and padding
    int newW = static_cast<int>(std::round(image.cols * info.ratio));
    int newH = static_cast<int>(std::round(image.rows * info.ratio));

    info.padW = (targetSize.width - newW) / 2;
    info.padH = (targetSize.height - newH) / 2;

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Add padding
    cv::Mat padded;
    cv::copyMakeBorder(
        resized, padded,
        info.padH, targetSize.height - newH - info.padH,
        info.padW, targetSize.width - newW - info.padW,
        cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114)
    );

    return padded;
}

cv::Mat YOLOInference::preprocess(const cv::Mat& image, PreprocessInfo& info) {
    // Letterbox resize
    cv::Mat resized = letterboxResize(image, inputSize_, info);

    // BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize
    cv::Mat normalized;
    rgb.convertTo(normalized, CV_32F);

    // Apply normalization
    std::vector<cv::Mat> channels(3);
    cv::split(normalized, channels);

    for (int i = 0; i < 3; i++) {
        channels[i] = (channels[i] - mean_[i]) / std_[i];
    }

    cv::merge(channels, normalized);

    return normalized;
}

std::vector<Detection> YOLOInference::predict(
    const cv::Mat& image,
    float confThreshold,
    float iouThreshold,
    int maxDet
) {
    if (image.empty()) {
        throw std::runtime_error("Empty image");
    }

    // Preprocess
    PreprocessInfo info;
    cv::Mat preprocessed = preprocess(image, info);

    // Prepare input tensor
    std::vector<float> inputData;
    inputData.resize(inputSize_.width * inputSize_.height * 3);

    // Convert HWC to CHW
    std::vector<cv::Mat> channels(3);
    cv::split(preprocessed, channels);

    size_t channelSize = inputSize_.width * inputSize_.height;
    for (int c = 0; c < 3; c++) {
        std::memcpy(
            inputData.data() + c * channelSize,
            channels[c].data,
            channelSize * sizeof(float)
        );
    }

    // Create input tensor
    std::vector<int64_t> inputShapeDims = {1, 3, inputSize_.height, inputSize_.width};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        inputData.data(),
        inputData.size(),
        inputShapeDims.data(),
        inputShapeDims.size()
    );

    // Run inference
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr},
        inputNames_.data(),
        &inputTensor,
        1,
        outputNames_.data(),
        outputNames_.size()
    );

    // Extract output data
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    size_t outputSize = std::accumulate(
        outputShape.begin(), outputShape.end(),
        1LL, std::multiplies<int64_t>()
    );

    std::vector<float> output(outputData, outputData + outputSize);

    // Postprocess based on task type
    if (taskType_ == "detect" || taskType_ == "detection") {
        return postprocessDetect(output, info, confThreshold, iouThreshold, maxDet);
    } else if (taskType_ == "segment" || taskType_ == "segmentation") {
        // Extract protos (second output)
        float* protosData = outputTensors[1].GetTensorMutableData<float>();
        auto protosShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();

        size_t protosSize = std::accumulate(
            protosShape.begin(), protosShape.end(),
            1LL, std::multiplies<int64_t>()
        );

        std::vector<float> protos(protosData, protosData + protosSize);

        return postprocessSegment(output, protos, info, confThreshold, iouThreshold, maxDet);
    } else {
        throw std::runtime_error("Unsupported task type for predict(): " + taskType_);
    }
}

std::vector<Detection> YOLOInference::postprocessDetect(
    const std::vector<float>& output,
    const PreprocessInfo& info,
    float confThreshold,
    float iouThreshold,
    int maxDet
) {
    // YOLOv8 output: [1, 84, 8400] -> [8400, 84]
    // Format: [cx, cy, w, h, class0_conf, class1_conf, ...]

    const int numClasses = 80;  // COCO classes
    const int numBoxes = 8400;

    std::vector<cv::Rect2f> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;

    for (int i = 0; i < numBoxes; i++) {
        // Find max class confidence
        float maxConf = 0.0f;
        int maxClassId = 0;

        for (int c = 0; c < numClasses; c++) {
            float conf = output[4 * numBoxes + c * numBoxes + i];
            if (conf > maxConf) {
                maxConf = conf;
                maxClassId = c;
            }
        }

        if (maxConf < confThreshold) {
            continue;
        }

        // Extract box
        float cx = output[0 * numBoxes + i];
        float cy = output[1 * numBoxes + i];
        float w = output[2 * numBoxes + i];
        float h = output[3 * numBoxes + i];

        // Convert to xyxy
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;

        boxes.emplace_back(x1, y1, w, h);
        scores.push_back(maxConf);
        classIds.push_back(maxClassId);
    }

    if (boxes.empty()) {
        return {};
    }

    // Apply NMS
    std::vector<int> keepIndices = nms(boxes, scores, iouThreshold);

    // Limit to maxDet
    if (static_cast<int>(keepIndices.size()) > maxDet) {
        keepIndices.resize(maxDet);
    }

    // Create detection results
    std::vector<Detection> detections;
    detections.reserve(keepIndices.size());

    for (int idx : keepIndices) {
        Detection det;
        det.box = scaleBox(boxes[idx], info);
        det.confidence = scores[idx];
        det.classId = classIds[idx];

        detections.push_back(det);
    }

    return detections;
}

std::vector<Detection> YOLOInference::postprocessSegment(
    const std::vector<float>& output,
    const std::vector<float>& protos,
    const PreprocessInfo& info,
    float confThreshold,
    float iouThreshold,
    int maxDet
) {
    // Similar to detection but with mask generation
    // For brevity, simplified implementation
    return postprocessDetect(output, info, confThreshold, iouThreshold, maxDet);
}

std::vector<PoseResult> YOLOInference::predictPose(
    const cv::Mat& image,
    float confThreshold,
    float iouThreshold,
    int maxDet
) {
    // Similar preprocessing and inference
    // Postprocess with keypoint extraction
    return {};  // Simplified
}

ClassificationResult YOLOInference::predictClassification(
    const cv::Mat& image,
    float confThreshold
) {
    // Simplified classification
    ClassificationResult result;
    result.classId = -1;
    result.confidence = 0.0f;
    return result;
}

std::vector<PoseResult> YOLOInference::postprocessPose(
    const std::vector<float>& output,
    const PreprocessInfo& info,
    float confThreshold,
    float iouThreshold,
    int maxDet
) {
    // Simplified
    return {};
}

ClassificationResult YOLOInference::postprocessClassification(
    const std::vector<float>& output,
    float confThreshold
) {
    // Simplified
    ClassificationResult result;
    result.classId = -1;
    result.confidence = 0.0f;
    return result;
}

std::vector<int> YOLOInference::nms(
    const std::vector<cv::Rect2f>& boxes,
    const std::vector<float>& scores,
    float iouThreshold
) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Sort by score (descending)
    std::sort(indices.begin(), indices.end(), [&scores](int a, int b) {
        return scores[a] > scores[b];
    });

    std::vector<int> keep;

    while (!indices.empty()) {
        int current = indices[0];
        keep.push_back(current);

        std::vector<int> remaining;

        for (size_t i = 1; i < indices.size(); i++) {
            int idx = indices[i];

            // Calculate IoU
            float intersectionArea = (boxes[current] & boxes[idx]).area();
            float unionArea = boxes[current].area() + boxes[idx].area() - intersectionArea;
            float iou = intersectionArea / unionArea;

            if (iou <= iouThreshold) {
                remaining.push_back(idx);
            }
        }

        indices = remaining;
    }

    return keep;
}

cv::Rect2f YOLOInference::scaleBox(const cv::Rect2f& box, const PreprocessInfo& info) {
    cv::Rect2f scaled = box;

    // Remove padding
    scaled.x -= info.padW;
    scaled.y -= info.padH;

    // Scale back
    scaled.x /= info.ratio;
    scaled.y /= info.ratio;
    scaled.width /= info.ratio;
    scaled.height /= info.ratio;

    // Clip to original image
    scaled.x = std::max(0.0f, std::min(scaled.x, static_cast<float>(info.originalSize.width)));
    scaled.y = std::max(0.0f, std::min(scaled.y, static_cast<float>(info.originalSize.height)));

    return scaled;
}

cv::Point2f YOLOInference::scalePoint(const cv::Point2f& point, const PreprocessInfo& info) {
    cv::Point2f scaled;
    scaled.x = (point.x - info.padW) / info.ratio;
    scaled.y = (point.y - info.padH) / info.ratio;
    return scaled;
}

cv::Mat YOLOInference::visualize(
    const std::vector<Detection>& detections,
    const cv::Mat& image,
    bool showLabels,
    bool showConf
) const {
    cv::Mat annotated = image.clone();

    int lineWidth = std::max(static_cast<int>(std::round((image.cols + image.rows) / 2 * 0.003)), 2);

    for (const auto& det : detections) {
        // Draw box
        cv::Scalar color = getColor(det.classId);
        cv::rectangle(annotated, det.box, color, lineWidth);

        // Draw label
        if (showLabels || showConf) {
            std::string label;
            if (showLabels) {
                label = getClassName(det.classId);
            }
            if (showConf) {
                if (!label.empty()) label += " ";
                label += cv::format("%.2f", det.confidence);
            }

            int baseLine;
            cv::Size textSize = cv::getTextSize(
                label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 1, &baseLine
            );

            cv::Point textOrg(det.box.x, det.box.y - 5);
            cv::rectangle(
                annotated,
                textOrg + cv::Point(0, baseLine),
                textOrg + cv::Point(textSize.width, -textSize.height - 5),
                color, -1
            );

            cv::putText(
                annotated, label, textOrg,
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1
            );
        }
    }

    return annotated;
}

cv::Mat YOLOInference::visualizePose(
    const std::vector<PoseResult>& poses,
    const cv::Mat& image,
    bool showConf
) const {
    // Simplified pose visualization
    return image.clone();
}

std::string YOLOInference::getClassName(int classId) const {
    if (classId >= 0 && classId < static_cast<int>(classNames_.size())) {
        return classNames_[classId];
    }
    return "class_" + std::to_string(classId);
}

cv::Scalar YOLOInference::getColor(int classId) const {
    // Generate consistent color for class ID
    cv::RNG rng(classId);
    return cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
}

} // namespace yolo
