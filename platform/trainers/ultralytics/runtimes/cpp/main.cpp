/**
 * @file main.cpp
 * @brief Example usage of YOLO ONNX Runtime wrapper
 */

#include "model_wrapper.h"
#include <iostream>
#include <exception>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <image.jpg>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string imagePath = argv[2];

    try {
        // Initialize model
        std::cout << "Loading model..." << std::endl;
        yolo::YOLOInference model(modelPath, "", true);

        // Load image
        std::cout << "Loading image..." << std::endl;
        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) {
            std::cerr << "Failed to load image: " << imagePath << std::endl;
            return 1;
        }

        std::cout << "Image size: " << image.size() << std::endl;

        // Run inference
        std::cout << "Running inference..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        auto detections = model.predict(image, 0.25f, 0.45f);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Detections: " << detections.size() << std::endl;
        std::cout << "Inference time: " << duration.count() << " ms" << std::endl;

        // Print results
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            std::cout << "[" << i << "] "
                      << model.getClassName(det.classId) << " "
                      << det.confidence << " "
                      << det.box << std::endl;
        }

        // Visualize
        std::cout << "Visualizing..." << std::endl;
        cv::Mat annotated = model.visualize(detections, image);

        std::string outputPath = "output.jpg";
        cv::imwrite(outputPath, annotated);

        std::cout << "Saved to: " << outputPath << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
