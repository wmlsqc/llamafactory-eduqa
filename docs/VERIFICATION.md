# 软件验证记录

验证日期：2026-09-15。环境：Windows、Python 3.12.14、项目独立虚拟环境。本次验证没有执行 7B GPU 训练或生成真实模型性能成绩。

## 已执行检查

| 检查 | 结果 | 说明 |
|---|---|---|
| 自动化测试 | **145 passed** | 数据/训练 36，评测/压测 48，服务 40，跨模块 2，部署 19 |
| 项目依赖 | `pip check` 通过 | 本机轻量应用/测试环境，无依赖冲突 |
| Python 源码 | 编译检查通过 | `python -m compileall -q src` |
| 云端 Shell | `bash -n` 通过 | 环境安装与完整实验脚本 |
| Docker Compose | 配置校验通过 | Docker Compose 2.29.2；未拉取并启动 GPU 镜像 |
| 实际数据处理 | 450/50/50 | 训练/验证/测试；规范化 prompt 无交叉，测试源含 8 个空白行并已记录 |
| 训练/合并/部署命令 | dry-run 通过 | 保存实际解析配置，未标为训练成功 |
| Web 界面 | Chromium 功能检查通过 | 多轮、SSE、停止、清空、错误恢复、HTML 转义及移动端布局；使用测试后端 |

测试运行命令：

```bash
python -m pytest -q --junitxml=artifacts/verification/pytest.xml
python -m pip check
python -m compileall -q src
bash -n scripts/bootstrap_cloud.sh
bash -n scripts/run_cloud_pipeline.sh
docker compose -f deploy/compose.yaml config --quiet
```

测试覆盖无效数据、相同问题冲突答案、数据泄漏、哈希篡改、输出保护、训练失败记录、错误基座合并、量化字段误用、认证、并发限制、SSE 使用量、两类断连路径、数学符号区分、评测参数一致性、截断回答、无 usage、进程组关闭和云端数据复用。

MockTransport 与 ASGI 测试回答仅存在 `tests/`。其延迟和分数不代表 GPU 模型；正式代码始终连接指定模型服务。当前测试有一条来自 Starlette/AnyIO 的弃用提示，不影响测试通过。

## 云端执行后补充的证据

完整项目实现包含云 GPU 执行路径，但需要在实际租用实例上运行后才能获得这些结果：

- 训练/导出的设备、驱动、软件版本、配置哈希与状态：`artifacts/runs/*/manifest.json`。
- 实际 LoRA 和合并权重：`artifacts/models/`。
- 同一 50 题测试集的模型回答和文本指标：`artifacts/eval/base/`、`artifacts/eval/tuned/`。
- 人工审核分数：各评测目录的 `blind_review.csv`。
- 并发 1/4 的 TTFT、延迟分位数和吞吐：`artifacts/benchmark/c1.json`、`c4.json`。

推荐租用配置为 RTX 4090 24GB，不会写入本机软件验证记录冒充训练设备。训练/推理依赖版本已按 LLaMA-Factory v0.9.3 与 vLLM v0.8.5 的上游约束交叉核对；云端环境完整安装、CUDA 执行和实际显存峰值仍以服务器运行记录为准。
