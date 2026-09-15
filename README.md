# 基于 LLaMA-Factory 的教育问答模型优化与部署

面向中学理科概念解释和解题答疑的完整工程项目：**Qwen2.5-7B-Instruct → 4-bit QLoRA 微调 → 独立评测 → LoRA 合并 → vLLM 推理 → 中文流式问答网页**。

项目提供可执行代码、两种云 GPU 训练配置、固定版本环境、自动化测试和完整云端运行脚本。训练和推理实际调用 LLaMA-Factory/vLLM；正式服务没有模拟回答，也没有预填性能成绩。

## 功能与产物

| 模块 | 实现 | 运行产物 |
|---|---|---|
| 数据处理 | 字段验证、NFKC 归一化、去重、冲突检查、训练/测试重复检测、固定种子划分 | train/val/test JSON、dataset_info、SHA-256 manifest |
| QLoRA 微调 | NF4 4-bit、BF16、LoRA rank 16、梯度累积/检查点、验证 loss 选优 | adapter、训练记录、解析后的 YAML、实际设备与版本 |
| 模型导出 | 检查基座一致性，重新加载非量化基座后合并 LoRA | 可供 vLLM 加载的完整模型 |
| 独立评测 | 基座/微调模型使用同测试集和生成参数，文本 F1/ROUGE-L、截断与失败统计 | predictions.jsonl、summary.json、Markdown 报告、人工评分表 |
| 并发压测 | SSE 首文本延迟、P50/P95、请求吞吐、服务端 token 用量，排除预热 | 逐请求记录与完整配置 JSON |
| Web 应用 | FastAPI 网关、中文多轮/流式界面、可选鉴权、并发限制、超时和断连清理 | 浏览器应用和 `/v1/chat/completions` |
| 云端运行 | 独立训练/推理环境、模型版本锁定、设备检查、完整实验脚本、Docker 方案 | 环境、训练、评测和部署记录 |

## 训练设备与参数

主配置：**云端 RTX 4090 24GB × 1、16 vCPU、64GB 内存、150GB SSD、Ubuntu 22.04、Python 3.11**。

可在提供对应规格的 [AutoDL 算力市场](https://www.autodl.com/market/list) 或 [RunPod RTX 4090 租用服务](https://www.runpod.io/gpu-models/rtx-4090) 选择实例，实时库存与价格以平台为准。备选 A100 40GB/80GB 配置已包含在 `configs/train_a100.yaml`。

| 参数 | 默认值 |
|---|---|
| 模型 | Qwen/Qwen2.5-7B-Instruct |
| 微调 | SFT + NF4 4-bit QLoRA，BF16 |
| LoRA | rank 16、alpha 32、dropout 0.05、target all |
| batch / 梯度累积 | 1 / 8 |
| 序列长度 / 轮数 | 2048 / 3 |
| 学习率 | 1e-4，cosine，warmup 0.1 |
| 数据 | 450 训练 / 50 验证 / 50 独立测试，seed 42 |
| 推理 | 合并后的非量化模型，vLLM，2048 上下文、最多 4 路序列 |

设备和参数属于项目运行配置；实际训练设备、时间、loss 和评测成绩由程序运行后记录。小规模样例用于工程学习，回答质量改善应以独立评测和人工核验为准。

## 快速开始

### 无 GPU 的软件检查

```bash
git clone https://github.com/wmlsqc/llamafactory-eduqa.git
cd llamafactory-eduqa
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest -q
eduqa prepare-data
eduqa train --dry-run
eduqa export --dry-run
eduqa start-vllm --dry-run
```

`dry-run` 只验证/生成配置，不会冒充训练成功。数据处理会真实读取随仓库提供的问答，生成固定划分和哈希记录。

### 云 GPU 上运行整个项目

在已租用的 Linux GPU 服务器中：

```bash
bash scripts/bootstrap_cloud.sh train
bash scripts/bootstrap_cloud.sh inference
bash scripts/run_cloud_pipeline.sh
```

完整脚本执行 **数据处理 → 模型下载 → 训练 → 合并 → 基座评测 → 微调评测 → 两组并发压测**，遇到失败会停止并保留运行记录。它不会发起服务器租用或付费操作。

首次建议按 [云端运行手册](docs/CLOUD_RUNBOOK.md) 先执行两步 GPU 训练检查，再运行完整实验。该手册还包含逐步命令、环境版本、显存调整、SSH 转发、Docker 部署与备份方式。

### 启动问答网页

完成权重合并后，在推理环境启动模型服务：

```bash
source .venv-inference/bin/activate
eduqa start-vllm
```

另开终端：

```bash
source .venv-inference/bin/activate
eduqa serve --host 127.0.0.1 --port 8080
```

访问 `http://127.0.0.1:8080`。云端访问可通过 SSH 将 8080 端口转发到本机。网页支持多轮对话、停止生成、清空和错误恢复。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"eduqa","messages":[{"role":"user","content":"为什么匀速直线运动不需要持续的合力？"}],"max_tokens":256,"stream":false}'
```

## 目录

```text
src/eduqa/               # 新实现的数据、训练、评测、压测、部署与应用
configs/                 # 4090/A100 QLoRA 配置及非量化合并配置
requirements/            # 训练/推理环境的固定版本
scripts/                 # 云端环境安装与完整实验入口
deploy/                  # 网关 Dockerfile 与 GPU Compose 配置
tests/                   # 单元、协议和跨模块测试，无真实模型夹具进入正式代码
docs/                    # 云端手册、数据说明、项目素材、验证记录、上游说明
EDU-QA/                  # 保留的上游原始代码与 500/50 条数据
artifacts/               # 本地生成的运行/模型/评测结果，不提交 Git
models/                  # 下载的基座模型，不提交 Git
```

## 评估与证据

默认评测集为独立的 50 个理科问题。字符 F1、ROUGE-L 和归一化精确匹配属于文本指标；数学推导、单位和条件完整性仍需人工评分。两个模型的比较必须使用相同数据哈希和生成参数，截断答案与请求失败分别记录。

压测仅采用真实 SSE 响应：忽略 role-only 帧，以首个非空内容作为 TTFT 终点；只有服务端返回 usage 时才计算 token 吞吐。测试传输层只用于自动化软件验证，不作为模型质量或 GPU 性能证据。

软件验证见 [VERIFICATION.md](docs/VERIFICATION.md)。仓库不附带未经运行的模型权重、训练耗时或性能提升数字；云端运行后的 `artifacts/` 是实验结果来源。

## 文档

- [云端设备、环境和完整运行步骤](docs/CLOUD_RUNBOOK.md)
- [数据来源、清洗与划分](docs/DATA_CARD.md)
- [项目介绍及简历素材](docs/PROJECT_DESCRIPTION.md)
- [验证范围和实际检查记录](docs/VERIFICATION.md)
- [上游原始说明](docs/UPSTREAM_README.md)

## 来源

本仓库 Fork 自 [shj-cfl/EDU-QA](https://github.com/shj-cfl/EDU-QA)，原始代码、数据和提交历史保留在 `EDU-QA/` 及上游历史中。新的 `src/eduqa` 工程实现将数据、QLoRA、合并、vLLM、评测与压测整合为统一流程，默认网页不调用旧版后端。

新增工程代码适用 [LICENSE-CODE](LICENSE-CODE)。上游 README 声明 MIT，但该基准提交没有独立 LICENSE，本仓库不代原作者补写。原始数据、模型权重和第三方依赖分别遵循各自来源条款。

训练框架：[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)；推理框架：[vLLM](https://github.com/vllm-project/vllm)；基座：[Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。
