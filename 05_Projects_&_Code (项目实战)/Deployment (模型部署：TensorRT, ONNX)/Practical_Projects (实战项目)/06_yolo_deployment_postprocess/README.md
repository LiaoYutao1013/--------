# 06 YOLO Letterbox + NMS 后处理验证器

对应知识点：YOLO 部署重点。

这个项目实现 YOLO 部署中最容易出错的两个环节：

- letterbox 缩放、padding 和坐标还原
- 置信度过滤与 NMS

同时输出 TensorRT dynamic shape profile 建议和 INT8 校准数据覆盖摘要。

## 运行

```bash
python main.py
```

输出文件：`artifacts/yolo_postprocess_report.json`。
