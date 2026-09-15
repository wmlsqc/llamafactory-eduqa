# 基于 LLaMA-Factory 的教育问答模型优化与部署

> EDU-QA 的学习复现 Fork：理科指令数据 → LoRA 监督微调 → 对话推理 → 基座与微调模型评测。

本仓库 Fork 自 [shj-cfl/EDU-QA](https://github.com/shj-cfl/EDU-QA)，保留上游代码、数据和提交历史。上游基准提交为 [`caa3685`](https://github.com/shj-cfl/EDU-QA/commit/caa3685d52fbb7e957ce1f3eb0f3e7ae1ca7434b)。本次整理新增中文项目导览、复现说明及简历能力对照；**尚未在本账号环境完成 GPU 训练、推理联调或效果复测**。

原作者介绍和环境说明见 [上游 README](docs/UPSTREAM_README.md)。本仓库不将上游实现或实验表现标为个人原创成果。

## 项目简介

面向物理、化学等理科学习场景，通过整理问答指令数据、引入知识三元组构造训练样本，使用 LLaMA-Factory 对 Qwen 系列模型进行 LoRA 监督微调。平台以 FastAPI 提供训练调度、模型加载、问答与评测接口，使用 Vue 3 展示训练状态、对话结果和基座/微调模型对比结果。

项目适合作为教育领域大模型微调与应用部署的复现起点。实际目标是学习数据构建、训练配置、模型推理及评估流程，是否改善回答质量应以独立测试和人工核验为准。

## 已有实现

| 环节 | 上游已有内容 | 主要文件 |
|---|---|---|
| 教育数据 | 500 条理科训练问答、50 条测试问答；上传数据和知识三元组转换 | `EDU-QA/data/`、后端 `/api/finetune` |
| 模型微调 | 调用 `llamafactory-cli train`，配置 SFT、LoRA、学习率、轮数、batch 和验证集比例 | `EDU-QA/EDU-FT/backend/main.py` |
| 状态展示 | 后台训练任务、训练状态查询、训练取消 | 同上及前端 `src/App.vue` |
| 对话推理 | Transformers 加载基座、PEFT 加载 LoRA，FastAPI 提供聊天接口 | 后端 `/api/load_model`、`/api/chat` |
| 效果评测 | 对比基座与 LoRA 的 BLEU、ROUGE-L、METEOR，并展示结果 | 后端 `/api/start_eval`、前端评测页面 |

技术栈：Python、PyTorch、LLaMA-Factory、Transformers、PEFT、FastAPI、Vue 3、Element Plus。

本次静态检查：后端 Python 文件语法解析通过；两份 JSONL 可解析且字段完整，训练和测试问题没有完全相同的文本。该检查不代表样本内容正确、没有语义泄漏或 GPU 功能已运行通过。

## 与项目方案的对应关系

| 简历方案中的能力 | 当前仓库状态 |
|---|---|
| 教育指令数据与模型监督微调 | 有相应数据和训练代码，需在自己的环境执行 |
| Qwen2.5-7B-Instruct 与 LoRA | 上游支持，模型权重需自行准备 |
| FastAPI 对话部署与基座/微调评测 | 有实现，需联调验证 |
| 4 bit QLoRA 训练 | **尚未实现**，当前训练入口未设置量化参数 |
| 合并 LoRA 后使用 vLLM 提供服务 | **尚未实现**，当前使用 Transformers/PEFT 推理 |
| 首 Token 延迟、并发吞吐和显存压测 | **尚未提供**，需要增加独立测试脚本 |

复现前可以将该项目写为“学习复现项目”或“项目方案”。训练和评测完成后，再依据自己的提交、日志和结果描述个人工作，不引用未实测的提升比例。

## 目录

```text
.
├── README.md                         # 本 Fork 的中文导览
├── docs/UPSTREAM_README.md            # 原作者说明，保留原文
└── EDU-QA/
    ├── data/
    │   ├── science_ft_500.jsonl       # 上游理科训练样本
    │   └── test.jsonl                 # 上游测试样本
    └── EDU-FT/
        ├── backend/
        │   ├── main.py               # 训练、推理、评测 API
        │   ├── evaluate_performance.py
        │   ├── radar.py
        │   └── requirements.txt
        └── frontend/                 # Vue 3 界面
```

## 复现步骤

推荐使用具有 CUDA GPU 的 Linux 环境。上游建议约 24 GB 显存，但具体需求随模型、序列长度、batch 和是否同时加载推理模型变化；本 Fork 未测试硬件下限。

### 1. 克隆并准备依赖

```bash
git clone https://github.com/wmlsqc/llamafactory-eduqa.git
cd llamafactory-eduqa
```

准备独立 Python 环境并安装与 CUDA 匹配的 PyTorch。安装 LLaMA-Factory 后，确认 `llamafactory-cli version` 可执行。后端还需 `transformers`、`peft`、`accelerate`、`jieba`、`nltk`、`rouge_chinese` 和 `requirements.txt` 中的依赖。前端使用支持 Vite 7 的 Node.js（20.19+ 或 22.12+）。

上游说明中的固定旧版 Transformers 与源码使用的 `dtype` 参数存在版本适配点，也未锁定 LLaMA-Factory 版本。应在调通后记录实际版本，不把上游安装命令视为已经验证的完整环境锁定文件。

### 2. 配置本机模型路径

准备 `Qwen2.5-7B-Instruct` 基座模型，将以下文件中的上游服务器路径替换为自己的模型路径：

- `EDU-QA/EDU-FT/frontend/src/App.vue`：模型选项和 `form.baseModel` 默认值。
- `EDU-QA/EDU-FT/backend/main.py`：`run_llama_factory` 的默认基座路径。

推理代码指定 `cuda:0`，需在 GPU 环境使用。暂不支持用 CPU 完整运行这条推理路径。

### 3. 启动应用

在仓库根目录打开后端终端：

```bash
cd EDU-QA/EDU-FT/backend
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

另开终端，从仓库根目录启动前端：

```bash
cd EDU-QA/EDU-FT/frontend
npm install
npm run dev
```

当前 `vite.config.js` 固定前端端口为 **5174**，前端 API 指向 `http://127.0.0.1:8001`。以上命令面向本机开发；远程服务器可通过 SSH 端口转发使用。

### 4. 数据、训练与评测

1. 在训练页面上传 `EDU-QA/data/science_ft_500.jsonl`，或添加知识三元组。后端生成自己的 `data/` 和 `dataset_info.json`。
2. 选择模型路径与 LoRA 配置，启动训练并保存日志、配置和适配器。
3. 等训练完成后，分别加载基座和 LoRA 进行对话测试；不要同时在显存不足的单卡上启动训练与推理。
4. 使用独立测试问题进行对比。BLEU、ROUGE-L、METEOR 是文本相似度指标，不能直接等同于理科答案正确率，需补充人工正确性检查。

后端的文件上传与训练接口没有身份认证，保留为本机实验工具使用；本次 Fork 不代表已完成公开服务部署加固。

## 下一步扩展

- 为 QLoRA 增加量化配置与独立 YAML，记录显存、训练配置和验证结果。
- 重新加载非量化基座后合并 LoRA，新增 vLLM 推理服务及请求示例。
- 增加固定输入长度和并发条件下的推理压测。
- 清洗样本、检查训练/测试重复与来源，记录模型回答的事实错误和概念遗漏。

## 来源与许可说明

原作者仓库：[shj-cfl/EDU-QA](https://github.com/shj-cfl/EDU-QA)。上游 README 声明采用 MIT，但在本次 Fork 的提交中未包含独立 `LICENSE` 文件。本仓库保留该声明与原始历史，不补写或伪造原作者的许可证。模型及数据仍以各自来源条款为准。

训练框架：[hiyouga/LlamaFactory](https://github.com/hiyouga/LlamaFactory)。
