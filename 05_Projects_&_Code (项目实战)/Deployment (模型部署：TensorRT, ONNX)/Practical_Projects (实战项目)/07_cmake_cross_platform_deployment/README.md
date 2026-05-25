# 07 CMake 跨平台部署模板

对应知识点：CMake 与跨平台视觉部署。

这个项目提供一个最小的 C++/CMake 视觉部署骨架，重点演示：

- `target-based` CMake 写法
- 头文件 / 源文件拆分
- `install()` 安装规则
- 跨平台构建入口

## 运行

```bash
cmake -S . -B build
cmake --build build
./build/vision_app
```

Windows 下可用：

```powershell
cmake -S . -B build
cmake --build build --config Release
.\build\Release\vision_app.exe
```

这个模板不依赖 OpenCV 或 TensorRT，适合先把 CMake 流程跑通，再替换成真实推理依赖。

当前仓库路径较深，构建时可能看到 object path 太长的警告。这本身就是跨平台部署里要处理的问题，推荐把源码或构建目录放到更短路径。

## Preset 方式

如果安装了 Ninja，也可以运行：

```bash
cmake --preset ninja-release
cmake --build --preset ninja-release
```

## Windows 深路径提示

如果 CMake 提示 object path 太长，可以把构建目录放到更短的位置：

```powershell
cmake -S . -B C:\build\vision_deploy
cmake --build C:\build\vision_deploy
```
