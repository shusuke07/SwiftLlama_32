// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "SwiftLlama",
    platforms: [
        .macOS(.v15),
        .iOS(.v16.4),
        .watchOS(.v11),
        .tvOS(.v16.4),
        .visionOS(.v2)
    ],
    products: [
        .library(name: "SwiftLlama", targets: ["SwiftLlama"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ggerganov/llama.cpp.git", branch: "master")
    ],
    targets: [
        .target(name: "SwiftLlama", 
                dependencies: [
                    "LlamaFramework"
                ]),
        .testTarget(name: "SwiftLlamaTests", dependencies: ["SwiftLlama"]),
        .binaryTarget(
            name: "LlamaFramework",
            url: "https://github.com/ggml-org/llama.cpp/releases/download/b6264/llama-b6264-xcframework.zip",
            checksum: "370b291e5b0785a1e3c186b6dea77e335e403f6a1c1f09013a5811c681278515"
        )
    ]
)
