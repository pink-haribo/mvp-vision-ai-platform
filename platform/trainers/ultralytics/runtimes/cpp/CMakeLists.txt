cmake_minimum_required(VERSION 3.15)
project(YOLOInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_DIR "" CACHE PATH "Path to ONNX Runtime directory")
if(NOT ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR to ONNX Runtime installation path")
endif()

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${ONNXRUNTIME_DIR}/lib
)

# Library
add_library(yolo_wrapper STATIC
    model_wrapper.cpp
)

target_link_libraries(yolo_wrapper
    ${OpenCV_LIBS}
    onnxruntime
)

# Example executable
add_executable(yolo_inference
    main.cpp
)

target_link_libraries(yolo_inference
    yolo_wrapper
    ${OpenCV_LIBS}
    onnxruntime
)

# Install
install(TARGETS yolo_wrapper yolo_inference
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES model_wrapper.h
    DESTINATION include
)
