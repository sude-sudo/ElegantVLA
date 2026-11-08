# 发布验证检查单

本清单用于区分已提供的仓库内容与尚需完成的发布验证，不替代测试报告。

## 仓库内容

- [x] 固定 GR00T 和 OpenPI 的上游 submodule revision。
- [x] 提供统一后端目录、共用调度模块和两套 integration 安装脚本。
- [x] 提供源码、启动器、配置及缓存路径测试。
- [x] 在文档中区分算法支持、后端专用优化和验证范围。

## 发布验证

- [ ] 审核 integration 与固定上游版本的差异及必要性。
- [ ] 从干净 clone 分别验证安装、baseline 推理和 ElegantVLA 推理。
- [ ] 固定论文复现配置，提供教师阈值、模型与 RL 权重的获取说明。
- [ ] 补齐成功率、延时和 FLOPs 的复现命令、计量口径及对应结果文件。
- [ ] 记录 GR00T 算子与 CUDA Graph 的数值一致性和 GPU 延时；OpenPI 的性能单独验证。
- [ ] 确认兼容包和可选分析工具的发布范围，检查未使用入口和临时调试配置。
- [ ] 对源码、配置和 Git 历史完成凭据、私人路径及不应发布文件的检查。
- [ ] 确认顶层许可证，核对第三方代码与模型权重许可，并添加 `LICENSE`。
- [ ] 添加论文引用、`CITATION.cff` 和模型、数据获取说明。
- [ ] 将无 GPU 的源码、启动器和配置检查接入 CI。

## 检查入口

以下命令从仓库根目录执行：

```bash
python -m unittest discover -s tests
python -m unittest discover -s elegantvla/openpi/tests -p test_config.py
bash scripts/install_gr00t_integration.sh --check
bash scripts/install_openpi_integration.sh --check
```

安装检查只核对上游版本和覆盖文件，不执行模型推理。
完整模型和 GPU 验证需使用相应后端的依赖环境，见 [GR00T](../elegantvla/gr00t/README.md) 与 [OpenPI](../elegantvla/openpi/README.md) 使用说明。
