# 云端设备与完整运行手册

## 设备配置

项目主配置选择 **1 × NVIDIA RTX 4090 24GB**。这是可租用的云端 GPU 规格；不是声称本仓库已经在该设备取得训练成绩。

| 资源 | 推荐配置 | 用途与理由 |
|---|---|---|
| GPU | RTX 4090 24GB × 1 | 支持 BF16；Qwen2.5-7B 采用 NF4 4-bit QLoRA，仅训练 LoRA 参数 |
| CPU | 16 vCPU | 数据预处理、tokenizer 和服务进程；分配不足时先减少 workers |
| 主机内存 | 64GB | 为数据预处理、非量化基座加载及 CPU 合并权重保留空间 |
| 数据盘 | 150GB SSD 起 | 基座、适配器、合并模型、两套 GPU Python 环境和缓存 |
| 系统 | Ubuntu 22.04，Python 3.11，Linux x86_64 | 与固定训练/推理依赖组合相配 |
| NVIDIA 驱动 | 建议 550.54.14 或更高 | PyTorch 2.6.0 与 vLLM 0.8.5 使用 CUDA 12.4 生态；以实际镜像兼容性为准 |

租用入口：[RunPod RTX 4090](https://www.runpod.io/gpu-models/rtx-4090)、[AutoDL 算力市场](https://www.autodl.com/market/list)、[AutoDL 帮助](https://www.autodl.com/docs/)。RunPod 公开页面列有 RTX 4090 24GB；AutoDL 可按 GPU 型号筛选，库存和价格以创建实例时为准。

备选为 **A100 40GB/80GB × 1**，使用 `--profile a100`。该配置将训练 batch 调整为 2、梯度累积为 4，保持单卡有效 batch 为 8。不要将 A100 的吞吐预期写成 4090 的实测结果。

费用按“实例单价 × 使用小时 + 持久磁盘/流量”计算。先执行 2 步训练检查，再决定完整训练时长；本仓库没有预设虚构耗时、租用价格或训练收益。本次代码编写没有发起租机、付费任务或 GPU 训练。

## 环境准备

租用后在服务器上运行；工作目录应放到持久数据盘。例如 AutoDL 可用 `/root/autodl-tmp`，其他平台按磁盘挂载位置调整。

```bash
git clone https://github.com/wmlsqc/llamafactory-eduqa.git
cd llamafactory-eduqa
bash scripts/bootstrap_cloud.sh train
bash scripts/bootstrap_cloud.sh inference
```

脚本分别创建 `.venv-train` 和 `.venv-inference`，避免 LLaMA-Factory 和 vLLM 的依赖相互覆盖。训练框架固定到 LLaMA-Factory `v0.9.3`；PyTorch `2.6.0`、Transformers `4.51.3`、PEFT `0.15.2`、bitsandbytes `0.45.5`。推理使用 vLLM `0.8.5`。安装成功后自动执行 `pip check`，并将实际依赖版本写入各环境的 `resolved-requirements.txt`。

不要另外安装最新版 Transformers 覆盖这些版本。镜像如果没有 `python3.11`，先安装 Python 3.11 或用 `PYTHON_BIN=/实际路径/python3.11` 指定解释器。

```bash
source .venv-train/bin/activate
eduqa doctor --require-gpu --min-vram-gb 24
eduqa prepare-data
eduqa download-model
```

模型默认下载到 `models/Qwen2.5-7B-Instruct`；下载程序先解析并记录 Hugging Face 模型提交 SHA，重试时继续使用已记录版本。访问 Hugging Face 如需令牌，可在服务器配置 `HF_TOKEN`，不要提交令牌。网络镜像可通过 `HF_ENDPOINT` 自行设置，使用前确认来源。

## 训练配置

| 参数 | RTX 4090 主配置 |
|---|---|
| 基座 | Qwen/Qwen2.5-7B-Instruct |
| 训练方式 | SFT + 4-bit QLoRA，bitsandbytes NF4 + double quantization |
| 模板 | qwen |
| LoRA | rank 16、alpha 32、dropout 0.05、target all |
| 序列长度 | 2048 |
| batch / 累积 | 1 / 8，单卡有效 batch 8 |
| 学习率 | 1e-4，cosine，warmup ratio 0.1 |
| 训练轮数 | 3 |
| 精度 / 显存措施 | BF16、gradient checkpointing、不使用 packing |
| 验证与保存 | 每 epoch，按验证 loss 选最佳，最多保留 2 个 checkpoint |
| 数据 | 450 条训练、50 条验证、50 条独立测试，seed 42 |

这些是配置值，不是质量提升结论。现有 500 条样例适合教学和工程验证，不代表大规模生产训练集。验证 loss 和独立题目表现可能不改善，结果应如实保留。

先检查生成配置，不启动训练：

```bash
eduqa train --profile 4090 --model models/Qwen2.5-7B-Instruct --dry-run
```

执行真实的 2 步 GPU 冒烟检查，使用独立输出目录：

```bash
eduqa train --profile 4090 --model models/Qwen2.5-7B-Instruct \
  --max-steps 2 --output-dir artifacts/models/smoke-adapter
```

检查日志和适配器产物后执行完整训练：

```bash
eduqa train --profile 4090 --model models/Qwen2.5-7B-Instruct
eduqa export --model models/Qwen2.5-7B-Instruct
```

训练写入 `artifacts/models/eduqa-adapter`，合并结果写入 `artifacts/models/eduqa-merged`。导出重新加载**非量化基座**，合并 LoRA 后生成完整权重；它不是 4-bit 量化部署文件。导出配置默认在 CPU 合并，故推荐 64GB 内存。对已有非空输出目录，程序默认拒绝覆盖；需要时显式指定新目录或 `--overwrite`。

每次训练/导出都保存独立的 `artifacts/runs/<时间与随机后缀>/manifest.json` 和 `resolved.yaml`。manifest 中的 GPU、驱动、软件版本和起止时间来自实际运行环境；失败也有记录。

## 基座与微调模型比较

同一张 4090 依次运行两个模型，避免同时占用显存。先退出训练进程并切换环境：

```bash
deactivate
source .venv-inference/bin/activate
eduqa start-vllm --model-dir models/Qwen2.5-7B-Instruct \
  --manifest artifacts/vllm/base.json
```

保持服务终端运行，等待终端显示服务启动完成。在第二个终端激活相同环境，先检查模型是否已加载（设置后端 key 时也会携带认证）：

```bash
source .venv-inference/bin/activate
python - <<'PY'
import json, os, urllib.request
key = os.getenv('EDUQA_BACKEND_API_KEY', '')
headers = {'Authorization': 'Bearer ' + key} if key else {}
req = urllib.request.Request('http://127.0.0.1:8000/v1/models', headers=headers)
with urllib.request.urlopen(req, timeout=10) as response:
    payload = json.load(response)
assert any(item['id'] == 'eduqa' for item in payload['data']), '模型尚未就绪'
print('eduqa ready')
PY
```

只有检查输出 `eduqa ready` 后再运行评测：

```bash
source .venv-inference/bin/activate
eduqa evaluate --label base --output artifacts/eval/base
```

在服务终端按 Ctrl+C 关闭基座服务，待显存释放，再启动合并模型：

```bash
eduqa start-vllm --model-dir artifacts/models/eduqa-merged \
  --manifest artifacts/vllm/tuned.json
```

合并模型启动后，同样先执行上述就绪检查，再在客户端终端运行：

```bash
eduqa evaluate --label tuned --output artifacts/eval/tuned \
  --compare-to artifacts/eval/base/summary.json
eduqa benchmark --requests 20 --concurrency 1 --output artifacts/benchmark/c1.json
eduqa benchmark --requests 20 --concurrency 4 --output artifacts/benchmark/c4.json
```

两组都使用 `temperature=0`、相同测试集和相同 `max_tokens`。比较程序核对测试集哈希与生成参数。文本 F1/ROUGE-L 只反映文本重合，还需填写 `blind_review.csv` 中的正确性、完整性、指令遵循评分。

压测在预热后计时，TTFT 从实际发起请求到第一个非空文本增量；token 数使用服务端 usage。缺少 usage 时不估算 token/s。默认问题较短，`max_tokens` 是输出上限而非固定生成长度；进行不同实验时要保持输入文本、输出上限和并发参数一致，并核对实际输出 token 分布。

## 网页与 API

模型服务运行于 `127.0.0.1:8000`。另外启动应用网关：

```bash
source .venv-inference/bin/activate
export EDUQA_BACKEND_URL=http://127.0.0.1:8000/v1
export EDUQA_MODEL=eduqa
eduqa serve --host 127.0.0.1 --port 8080
```

浏览器打开 `http://127.0.0.1:8080` 可进行多轮、流式问答。网关提供 `/healthz`、`/readyz` 和 `/v1/chat/completions`，支持可选 Bearer key、参数约束、并发限制及断连清理。

从个人电脑访问云端，使用租用平台显示的 SSH 地址与端口建立转发：

```bash
ssh -p <SSH端口> -L 8080:127.0.0.1:8080 <用户>@<服务器地址>
```

再在个人电脑打开 `http://127.0.0.1:8080`。需要认证时在服务器设置 `EDUQA_API_KEY`，网页支持输入访问 key；后端认证使用 `EDUQA_BACKEND_API_KEY`，不要把它暴露给浏览器。

Docker 备选方式（需支持 GPU device reservations 的 Docker Compose v2 与 NVIDIA Container Toolkit，使用绝对模型目录）：

```bash
export EDUQA_MODEL_PATH="$(pwd)/artifacts/models/eduqa-merged"
docker compose -f deploy/compose.yaml up --build
```

## 一次运行全部实验

两个环境已安装好、没有占用 8000 端口、正式模型和评测输出目录未被先前实验占用时，可以执行：

```bash
bash scripts/run_cloud_pipeline.sh
```

它实际执行数据处理、下载、训练、合并、基座评测、微调评测和两组压测；数据已存在时先核对 manifest/hash 后复用，任一步失败即停止，退出时关闭自身启动的模型服务。它不会租用服务器，不会生成预填成绩。已执行过正式训练/评测的阶段可按上方手动命令续接，避免覆盖历史实验。

## 常见问题

- **OOM**：确认训练和推理没有并存；先用 2048 序列长度、batch 1。仍不足时将训练 cutoff 改为 1024，或调低 vLLM `--max-model-len` / `--max-num-seqs`，并记录配置差异。
- **模型找不到**：确认 download 完成，训练与导出都传同一个 `--model`；导出会检查适配器记录的基座。
- **合并量化模型失败**：必须重新加载原始基座，导出 YAML 中不能带 quantization 参数。
- **网关 503**：先查看模型服务日志、`/v1/models` 与配置模型名，`/healthz` 正常不代表 GPU 模型就绪。
- **评测/压测失败**：失败请求会记录并令 CLI 返回非零，不将失败算作成功；排查后使用新的输出目录重新运行。
- **服务器停止计费**：备份 adapter、merged model、runs、eval 和 benchmark，再停止或释放计算实例；持久存储是否继续计费以平台规则为准。
