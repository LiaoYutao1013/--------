# 05 部署就绪审计器

对应知识点：部署检查清单。

这个项目把部署前检查清单做成一个可运行的审计脚本。脚本会读取 `deployment_manifest.json`，检查预处理、后处理、一致性误差、动态输入、性能记录和异常输入处理。

## 运行

```bash
python main.py
```

你也可以审计自己的部署清单：

```bash
python main.py --manifest deployment_manifest.json
```

输出文件：`artifacts/readiness_report.json`。
