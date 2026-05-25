# CMake 跨平台构建与视觉部署

CMake 是视觉任务走向跨平台部署时最常见的构建系统之一。它本身不是编译器，而是用来生成不同平台上的构建文件：

- Windows: Visual Studio / Ninja / MSBuild
- Linux: Make / Ninja
- macOS: Xcode / Ninja

当你的视觉任务从 Python 原型走向 C++ 推理程序时，CMake 通常负责：

- 组织源文件、头文件和资源文件。
- 查找 OpenCV、ONNX Runtime、TensorRT、CUDA 等依赖。
- 统一 Debug / Release 构建。
- 打包可执行文件和动态库。
- 支持跨平台部署和交叉编译。

## 推荐写法

优先使用 target-based CMake，而不是全局变量式写法：

```cmake
cmake_minimum_required(VERSION 3.20)
project(vision_deploy LANGUAGES CXX)

add_executable(vision_app
    src/main.cpp
    src/inference.cpp
)

target_compile_features(vision_app PRIVATE cxx_std_17)
target_include_directories(vision_app PRIVATE include)
target_link_libraries(vision_app PRIVATE opencv_core opencv_imgcodecs)
```

优点：

- 依赖关系清晰。
- 目标之间不容易互相污染。
- 更适合大型视觉工程。

## 目录组织建议

```text
project/
  CMakeLists.txt
  cmake/
    toolchains/
    modules/
  include/
  src/
  third_party/
  assets/
  tests/
```

## 视觉部署常见依赖

### OpenCV

用于：

- 图像读取和预处理。
- 视频流处理。
- 后处理和可视化。

```cmake
find_package(OpenCV REQUIRED)
target_link_libraries(vision_app PRIVATE ${OpenCV_LIBS})
target_include_directories(vision_app PRIVATE ${OpenCV_INCLUDE_DIRS})
```

### ONNX Runtime

用于跨平台推理。常见问题是：

- 库路径如何找到。
- 头文件和运行时库如何配置。
- Windows 下 DLL 是否放在可执行文件旁边。

```cmake
find_package(onnxruntime REQUIRED)
target_link_libraries(vision_app PRIVATE onnxruntime)
```

### TensorRT

TensorRT 常用于 NVIDIA GPU 推理。构建时需要处理：

- CUDA 版本匹配。
- TensorRT include/lib 路径。
- FP16 / INT8 相关运行依赖。

TensorRT 项目通常会额外配置：

- `CUDA_TOOLKIT_ROOT_DIR`
- `TENSORRT_ROOT`
- `CUDA_ARCHITECTURES`

## 示例：最小跨平台 CMakeLists

```cmake
cmake_minimum_required(VERSION 3.20)
project(vision_deploy LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

add_library(vision_core
    src/inference.cpp
    src/postprocess.cpp
)

target_include_directories(vision_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(OpenCV REQUIRED)
target_link_libraries(vision_core
    PUBLIC
        ${OpenCV_LIBS}
)

add_executable(vision_app src/main.cpp)
target_link_libraries(vision_app PRIVATE vision_core)

install(TARGETS vision_core vision_app
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
```

## 关键命令

```bash
cmake -S . -B build
cmake --build build --config Release
cmake --install build --prefix install
```

常用生成器：

- `-G Ninja`
- `-G "Visual Studio 17 2022"`

Windows 路径较深时，CMake 可能提示 object path 太长。视觉工程目录本身经常很深，建议：

- 把源码放在较短路径，例如 `D:/cv/vision_app`。
- 或把构建目录放到短路径，例如 `cmake -S . -B C:/build/vision_app`。
- 开启 Windows 长路径支持只能缓解一部分工具链问题，仍建议保持工程路径短。

## Debug / Release

部署时大多关心 Release：

- `Debug`：带符号，方便调试。
- `Release`：优化开启，适合交付。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

注意：在 Visual Studio 这类多配置生成器下，配置类型通常在构建阶段选择，而不是只靠 `CMAKE_BUILD_TYPE`。

## 运行时库问题

跨平台部署最容易出错的是运行时依赖：

- Windows: `.exe` 旁边要有 `.dll`
- Linux: 需要 `LD_LIBRARY_PATH` 或 rpath
- macOS: 需要 `@rpath` / `@loader_path`

建议：

- 通过 `install()` 打包完整运行目录。
- 记录第三方库版本。
- 把模型文件、配置文件和字典文件一起安装。

## 交叉编译

当你把视觉程序部署到 ARM、Jetson 或嵌入式 Linux 时，往往需要 toolchain 文件：

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
```

然后：

```bash
cmake -S . -B build-arm -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/aarch64.cmake
```

## CMakePresets

`CMakePresets.json` 可以把常用构建配置写进项目，减少命令差异：

```json
{
  "version": 3,
  "configurePresets": [
    {
      "name": "ninja-release",
      "generator": "Ninja",
      "binaryDir": "${sourceDir}/build/ninja-release",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Release"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "ninja-release",
      "configurePreset": "ninja-release"
    }
  ]
}
```

使用：

```bash
cmake --preset ninja-release
cmake --build --preset ninja-release
```

## 推荐实践

1. 先让一个最小 `vision_app` 在本机跑通。
2. 再接入 OpenCV 处理图像输入。
3. 再接入 ONNX Runtime 或 TensorRT 推理。
4. 再补 `install()`、打包脚本和运行时库拷贝。
5. 最后做交叉编译和平台适配。

## 示例目录模板

```text
vision_deploy/
  CMakeLists.txt
  include/
    vision/
      inference.hpp
      postprocess.hpp
  src/
    main.cpp
    inference.cpp
    postprocess.cpp
  models/
    model.onnx
  configs/
    deploy.json
```

## 与视觉任务的关系

CMake 是“工程落地层”，不是算法层。它特别适合：

- 分类、检测、分割、姿态、OCR 的 C++ 推理程序。
- 统一 OpenCV / ONNX Runtime / TensorRT 的接入。
- 跨平台部署到 Windows、Linux、Jetson、工控机。

如果你的视觉任务最终要从 Python 原型交付成可执行程序，CMake 基本绕不开。
