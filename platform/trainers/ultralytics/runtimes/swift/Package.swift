// swift-tools-version:5.5
import PackageDescription

let package = Package(
    name: "YOLOInference",
    platforms: [
        .iOS(.v13),
        .macOS(.v10_15)
    ],
    products: [
        .library(
            name: "YOLOInference",
            targets: ["YOLOInference"]
        ),
    ],
    targets: [
        .target(
            name: "YOLOInference",
            dependencies: [],
            path: ".",
            sources: ["ModelWrapper.swift"]
        ),
    ]
)
