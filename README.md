# 🎓 EduQA: 知识图谱驱动的科学教育大模型指令微调平台

![Vue3](https://img.shields.io/badge/Frontend-Vue3-4FC08D?style=flat-square&logo=vue.js)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688?style=flat-square&logo=fastapi)
![LLaMA-Factory](https://img.shields.io/badge/Engine-LLaMA--Factory-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

EduQA 是一个面向垂直教育领域的开箱即用的大语言模型（LLM）定制平台。旨在通过注入结构化的知识图谱专家知识，解决通用大模型在专业理科问答中易产生“事实性幻觉”的痛点。

本项目基于前后端分离架构设计，底层集成 LLaMA-Factory ，包含**数据构建 -> LoRA 微调 -> 对话测试 -> 定量评测**的完整生命周期。


## ⚙️ Architecture

* **前端 (Frontend)**: Vue3 + Element-Plus + ECharts + Axios
* **后端核心 (Backend)**: FastAPI + Uvicorn + Python 3.11
* **大模型生态 (AI Stack)**: PyTorch + Hugging Face (`transformers`, `peft`)
* **训练引擎**: [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
* **推荐基座模型**: Qwen2.5-7B-Instruct / Yi-1.5-6B-Chat

---

## 🛠️ Environments

### 1. 基础环境准备
系统要求：Linux，单卡显存 $\ge$ 24GB（建议多卡避免系统功能中模型载入冲突）。
```bash
# 1. 创建 Python 3.11 环境
conda create -n edu_ft python=3.11 -y
conda activate edu_ft
```

### 2. 安装核心依赖

```bash
# 安装 PyTorch (根据的 CUDA 版本调整)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装与微调兼容的核心大模型组件
pip install transformers==4.40.2 peft==0.11.1 accelerate
```

### 3. 安装 LLaMA-Factory 训练引擎
```bash
# 将 LLaMA-Factory 拉取至同级目录
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 4. 安装后端依赖
```bash
pip install fastapi uvicorn pydantic python-multipart
pip install nltk rouge_chinese jieba
```

### 5. 模型准备
从 Hugging Face 或 ModelScope 下载基座模型权重（以 `Qwen2.5-7B-Instruct` 为例），并记录其在服务器上的绝对路径。

---

## 🚀 Quick Start

### 启动后端服务 (Backend)
进入后端项目目录，直接运行主脚本：
```bash
cd backend
python main.py
```
> 后端服务默认运行在：`http://0.0.0.0:8001`

### 启动前端界面 (Frontend)
进入前端项目目录：
```bash
cd frontend

# 安装 Node 依赖 (首次运行)
npm install

# 启动本地开发服务器
npm run dev
```
> 访问终端输出的本地地址（通常为 `http://localhost:5173`）即可进入系统。

---

## 📂 目录结构简述

```text
EduQA/
├── backend/
│   ├── main.py                # FastAPI 后端核心主程序 (处理调度、加载、评测逻辑)
│   ├── data/                  # 用户上传的原始数据集与合成指令数据存放处
│   │   └── dataset_info.json  # 数据集与 LLaMA-Factory 的动态注册映射文件
│   └── saves/                 # 微调生成的 LoRA 权重与评测分数持久化目录
├── frontend/
│   ├── src/
│   │   ├── App.vue            # 前端主页面 (整合微调中枢、对话测试、定量测评)
│   │   └── main.js
│   ├── package.json
│   └── vite.config.js
└── README.md
```

## 🤝 贡献与开源协议

本项目作为本科毕业设计开源发布，采用 [MIT License](LICENSE) 协议。欢迎提交 Issue 或 Pull Request，一起探索生成式 AI 在教育垂直领域的无限可能！

**致谢**：感谢 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) 团队提供的出色且易用的微调训练引擎。
