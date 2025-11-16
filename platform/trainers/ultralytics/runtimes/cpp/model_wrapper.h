/**
 * @file model_wrapper.h
 * @brief ONNX Runtime C++ wrapper for YOLO models
 *
 * Provides easy-to-use C++ interface for running YOLO models exported to ONNX format.
 *
 * Requirements:
 *   - ONNXRuntime (https://github.com/microsoft/onnxruntime)
 *   - OpenCV 4.x
 *   - nlohmann/json (https://github.com/nlohmann/json)
 *
 * Build:
 *   g++ -std=c++17 -O3 model_wrapper.cpp main.cpp \
 *       -I/path/to/onnxruntime/include \
 *       -L/path/to/onnxruntime/lib -lonnxruntime \
 *       `pkg-config --cflags --libs opencv4` \
 *       -o yolo_inference
 *
 * Usage:
 *   YOLOInference model("model.onnx", "metadata.json");
 *   auto results = model.predict("image.jpg", 0.25f, 0.45f);
 *   cv::Mat annotated = model.visualize(results, "image.jpg");
 *   cv::imwrite("output.jpg", annotated);
 */

#ifndef YOLO_MODEL_WRAPPER_H
#define YOLO_MODEL_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>

namespace yolo {

/**
 * @brief Detection result for a single object
 */
struct Detection {
    cv::Rect2f box;        ///< Bounding box (x, y, width, height)
    float confidence;      ///< Confidence score
    int classId;          ///< Class ID
    cv::Mat mask;         ///< Instance mask (for segmentation)
};

/**
 * @brief Pose estimation result
 */
struct PoseResult {
    cv::Rect2f box;                      ///< Bounding box
    float confidence;                    ///< Confidence score
    std::vector<cv::Point2f> keypoints;  ///< Keypoint positions (x, y)
    std::vector<float> kptConfidences;   ///< Keypoint confidences
};

/**
 * @brief Classification result
 */
struct ClassificationResult {
    int classId;                    ///< Predicted class ID
    float confidence;               ///< Confidence score
    std::vector<float> probabilities; ///< All class probabilities
};

/**
 * @brief Preprocessing information for inverse transform
 */
struct PreprocessInfo {
    cv::Size originalSize;  ///< Original image size
    float ratio;           ///< Scale ratio
    int padW;              ///< Left/right padding
    int padH;              ///< Top/bottom padding
};

/**
 * @brief YOLO inference engine using ONNX Runtime
 */
class YOLOInference {
public:
    /**
     * @brief Constructor
     * @param modelPath Path to ONNX model file
     * @param metadataPath Path to metadata.json (optional, auto-detected if empty)
     * @param useGPU Enable GPU acceleration (default: true)
     */
    YOLOInference(
        const std::string& modelPath,
        const std::string& metadataPath = "",
        bool useGPU = true
    );

    /**
     * @brief Destructor
     */
    ~YOLOInference();

    /**
     * @brief Run object detection
     * @param image Input image (BGR format)
     * @param confThreshold Confidence threshold (0.0-1.0)
     * @param iouThreshold IoU threshold for NMS (0.0-1.0)
     * @param maxDet Maximum number of detections
     * @return Vector of detections
     */
    std::vector<Detection> predict(
        const cv::Mat& image,
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDet = 300
    );

    /**
     * @brief Run pose estimation
     * @param image Input image
     * @param confThreshold Confidence threshold
     * @param iouThreshold IoU threshold for NMS
     * @param maxDet Maximum number of detections
     * @return Vector of pose results
     */
    std::vector<PoseResult> predictPose(
        const cv::Mat& image,
        float confThreshold = 0.25f,
        float iouThreshold = 0.45f,
        int maxDet = 300
    );

    /**
     * @brief Run image classification
     * @param image Input image
     * @param confThreshold Confidence threshold
     * @return Classification result
     */
    ClassificationResult predictClassification(
        const cv::Mat& image,
        float confThreshold = 0.25f
    );

    /**
     * @brief Visualize detection results
     * @param detections Detection results
     * @param image Original image
     * @param showLabels Show class labels
     * @param showConf Show confidence scores
     * @return Annotated image
     */
    cv::Mat visualize(
        const std::vector<Detection>& detections,
        const cv::Mat& image,
        bool showLabels = true,
        bool showConf = true
    ) const;

    /**
     * @brief Visualize pose results
     * @param poses Pose results
     * @param image Original image
     * @param showConf Show confidence scores
     * @return Annotated image
     */
    cv::Mat visualizePose(
        const std::vector<PoseResult>& poses,
        const cv::Mat& image,
        bool showConf = true
    ) const;

    /**
     * @brief Get class name by ID
     * @param classId Class ID
     * @return Class name (or "class_N" if not found)
     */
    std::string getClassName(int classId) const;

    /**
     * @brief Get task type
     * @return Task type string
     */
    std::string getTaskType() const { return taskType_; }

private:
    // Initialization
    void loadMetadata(const std::string& metadataPath);
    void initializeSession(const std::string& modelPath, bool useGPU);

    // Preprocessing
    cv::Mat preprocess(const cv::Mat& image, PreprocessInfo& info);
    cv::Mat letterboxResize(const cv::Mat& image, cv::Size targetSize, PreprocessInfo& info);

    // Postprocessing
    std::vector<Detection> postprocessDetect(
        const std::vector<float>& output,
        const PreprocessInfo& info,
        float confThreshold,
        float iouThreshold,
        int maxDet
    );

    std::vector<Detection> postprocessSegment(
        const std::vector<float>& output,
        const std::vector<float>& protos,
        const PreprocessInfo& info,
        float confThreshold,
        float iouThreshold,
        int maxDet
    );

    std::vector<PoseResult> postprocessPose(
        const std::vector<float>& output,
        const PreprocessInfo& info,
        float confThreshold,
        float iouThreshold,
        int maxDet
    );

    ClassificationResult postprocessClassification(
        const std::vector<float>& output,
        float confThreshold
    );

    // Utility functions
    std::vector<int> nms(
        const std::vector<cv::Rect2f>& boxes,
        const std::vector<float>& scores,
        float iouThreshold
    );

    cv::Rect2f scaleBox(const cv::Rect2f& box, const PreprocessInfo& info);
    cv::Point2f scalePoint(const cv::Point2f& point, const PreprocessInfo& info);
    cv::Scalar getColor(int classId) const;

    // ONNX Runtime members
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::vector<int64_t> inputShape_;

    // Metadata
    nlohmann::json metadata_;
    std::string taskType_;
    std::vector<std::string> classNames_;
    cv::Size inputSize_;
    std::vector<float> mean_;
    std::vector<float> std_;
};

} // namespace yolo

#endif // YOLO_MODEL_WRAPPER_H
