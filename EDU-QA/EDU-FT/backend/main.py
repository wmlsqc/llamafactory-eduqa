from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
import os
import subprocess
from datetime import datetime
import json
import torch
import torch.nn as nn
import jieba
import gc
import types
import nltk
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel

# 告诉 NLTK 明确的本地寻找路径
nltk.data.path.append(os.path.expanduser('~/nltk_data'))
active_training_process = None
app = FastAPI(title="EduQA 全链路微调与推理平台")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data", exist_ok=True)
os.makedirs("saves", exist_ok=True)

# ================= 全局状态管理器 =================
chat_model = None
chat_tokenizer = None
eval_status_db = {"status": "idle", "results": None, "message": "", "progress": 0}
training_status_db = {"status": "idle", "step": [], "loss": [], "progress": 0, "current_loss": 0}

# ================= 微调训练模块 =================

def run_llama_factory(config_data: dict):
    """
    真正拉起后台 LLaMA Factory 训练的进程（最终进化版）
    整合了：1.显卡动态对齐 2.进程句柄捕获 3.状态码兼容性 4.多卡死锁防护
    """
    global training_status_db, active_training_process
    print(f"\n🚀 开始后台微调任务...")
    
    # 1. 路径与配置准备
    base_model = config_data.get("baseModel", "/home/common/hjshen_2025/projects/Qwen/Qwen2.5-7B-Instruct")
    model_name = base_model.split('/')[-1]
    output_dir = f"saves/{model_name}_lora_web" 
    log_file_path = f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    # 初始化状态
    training_status_db["status"] = "training"
    training_status_db["progress"] = 0
    training_status_db["message"] = "训练正在初始化..."
    
    dataset_name = config_data.get("dataset_key", "merged_dataset")
    template_name = "qwen" if "qwen" in base_model.lower() else "default"
    
    # 2. 组装指令
    cmd = [
        "llamafactory-cli", "train",
        "--stage", "sft",
        "--do_train",
        "--model_name_or_path", base_model,
        "--dataset_dir", "data", 
        "--dataset", dataset_name,
        "--template", template_name,
        "--finetuning_type", config_data.get("finetuneType", "lora"),
        "--output_dir", output_dir,
        "--learning_rate", str(config_data.get("lr", "1e-4")),
        "--num_train_epochs", str(config_data.get("epochs", 5)),
        "--per_device_train_batch_size", str(config_data.get("batchSize", 1)),
        "--val_size", str(config_data.get("valSize", 5) / 100),
        "--logging_steps", "2",
        "--trust_remote_code",
        "--overwrite_output_dir",
        "--fp16" 
    ]
    
    if config_data.get("finetuneType") == "lora":
        cmd.extend([
            "--lora_rank", str(config_data.get("loraRank", 8)),
            "--lora_alpha", str(config_data.get("loraAlpha", 16))
        ])
    
    # 3. 显卡环境军管（继承命令行指定的显卡）
    env = os.environ.copy()
    current_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
    
    if current_gpu:
        print(f"🕵️ 检测到系统已指定显卡: {current_gpu}，微调任务将自动对齐。")
    else:
        print("⚠️ 未检测到显卡环境，默认强制使用物理 0 号卡，防止多卡死锁。")
        env["CUDA_VISIBLE_DEVICES"] = "0"
    
    # 核心：强制单进程模式，防止 DDP 引起的 8 卡互相等待死锁
    env["WORLD_SIZE"] = "1" 

    try:
        # 4. 开启日志并拉起进程
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            # 🚨 将进程对象赋值给全局变量，供 cancel_finetune 接口调用
            active_training_process = subprocess.Popen(
                cmd, 
                stdout=log_file, 
                stderr=subprocess.STDOUT, 
                text=True, 
                env=env
            )
            
            print(f"🔥 进程已启动 (PID: {active_training_process.pid})，正在炼丹...")
            active_training_process.wait() # 阻塞等待训练结束
            
        # 5. 训练结束后的状态结算
        if active_training_process and active_training_process.returncode == 0:
            print("\n✅ 后台微调任务顺利完成！")
            training_status_db["status"] = "completed" 
            training_status_db["progress"] = 100
        elif active_training_process and active_training_process.returncode == -15:
            # -15 通常是 SIGTERM，即用户手动点击了取消
            print("\n🛑 训练任务已被用户手动中止。")
            training_status_db["status"] = "idle"
        else:
            print(f"\n❌ 微调任务非正常退出，请查看日志: {log_file_path}")
            training_status_db["status"] = "error"
            
    except Exception as e:
        print(f"\n🚨 致命错误：后台线程发生崩溃: {str(e)}")
        training_status_db["status"] = "error"
        training_status_db["message"] = str(e)
    finally:
        # 无论成功失败，训练结束后清空全局句柄
        active_training_process = None



@app.post("/api/cancel_finetune")
async def cancel_finetune():
    global active_training_process, training_status_db
    
    if active_training_process and active_training_process.poll() is None:
        # 🚨 暴力终止进程及其子进程
        active_training_process.terminate() 
        training_status_db["status"] = "idle"
        training_status_db["message"] = "训练已手动取消"
        print("🛑 收到取消指令，微调任务已中止。")
        return {"status": "success", "message": "训练已成功中止"}
    
    return {"status": "error", "message": "当前没有正在运行的训练任务"}


@app.post("/api/finetune")
async def trigger_finetuning(
    background_tasks: BackgroundTasks,
    config: str = Form(...),
    manual_triples: str = Form("[]"),
    file: Optional[UploadFile] = File(None)
):
    global training_status_db
    config_data = json.loads(config)
    triples_data = json.loads(manual_triples)
    merged_data = []

    actual_filename = file.filename if (file and file.filename) else "custom_triples.jsonl"
    dataset_key = actual_filename.rsplit('.', 1)[0] 

    if file and file.filename:
        content = await file.read()
        lines = content.decode("utf-8").strip().split('\n')
        for line in lines:
            if line.strip():
                try: merged_data.append(json.loads(line))
                except Exception: pass

    for t in triples_data:
        if t.get("sub") and t.get("rel") and t.get("obj"):
            merged_data.append({
                "instruction": f"请结合学科知识解释一下，{t['sub']}的{t['rel']}是什么？",
                "input": "",
                "output": f"关于{t['sub']}的{t['rel']}，具体表现为：{t['obj']}。"
            })

    if not merged_data:
        return {"status": "error", "message": "训练集为空！请上传文件或添加知识三元组。"}

    save_path = os.path.join("data", actual_filename)
    with open(save_path, "w", encoding="utf-8") as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 🚨 猛药 6：动态更新后端 data 目录下的户口本，并映射正确字段
    info_path = "data/dataset_info.json"
    info = json.load(open(info_path, "r", encoding="utf-8")) if os.path.exists(info_path) else {}
    
    info[dataset_key] = {
        "file_name": actual_filename,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output"
        }
    }
    
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    config_data["dataset_key"] = dataset_key 

    training_status_db = {"status": "training", "step": [], "loss": [], "progress": 0, "current_loss": 0}
    background_tasks.add_task(run_llama_factory, config_data)
    
    return {"status": "success", "message": f"数据集 '{actual_filename}' 注册完毕！开始微调模型。"}

@app.get("/api/training_status")
async def get_training_status(base_model: str):
    global training_status_db
    output_dir = f"saves/{base_model.split('/')[-1]}_lora_web"
    log_jsonl = os.path.join(output_dir, "trainer_log.jsonl")
    state_file = os.path.join(output_dir, "trainer_state.json")
    
    try:
        # ================= 优先雷达：读取 LLaMA-Factory 的实时日志 =================
        if os.path.exists(log_jsonl):
            steps, losses = [], []
            total_steps = 100
            with open(log_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    data = json.loads(line)
                    if "loss" in data and "current_steps" in data:
                        steps.append(data["current_steps"])
                        losses.append(data["loss"])
                        if "total_steps" in data:
                            total_steps = data["total_steps"]
            
            if steps:
                training_status_db["step"] = steps
                training_status_db["loss"] = losses
                training_status_db["current_loss"] = losses[-1]
                # 根据当前步数和总步数计算实时进度百分比
                if training_status_db["status"] == "training":
                    training_status_db["progress"] = min(99, int((steps[-1] / total_steps) * 100)) if total_steps > 0 else 0
                    
        # ================= 兜底雷达：读取 Hugging Face 的最终结算文件 =================
        elif os.path.exists(state_file):
            state_data = json.load(open(state_file, "r", encoding="utf-8"))
            steps, losses = [], []
            for log in state_data.get("log_history", []):
                if "loss" in log and "step" in log:
                    steps.append(log["step"])
                    losses.append(log["loss"])
            if steps:
                training_status_db["step"] = steps
                training_status_db["loss"] = losses
                training_status_db["current_loss"] = losses[-1]
            if training_status_db["status"] == "training":
                max_steps = state_data.get("max_steps", 100)
                training_status_db["progress"] = min(99, int((state_data.get("global_step", 0) / max_steps) * 100)) if max_steps > 0 else 0
    except Exception as e:
        print(f"⚠️ 读取训练状态出错: {e}")
        
    return training_status_db

# ================= 加载与对话模块 =================

@app.get("/api/check_model")
async def check_model(base_model: str):
    lora_path = f"saves/{base_model.split('/')[-1]}_lora_web"
    return {"has_lora": os.path.exists(lora_path) and os.path.exists(os.path.join(lora_path, "adapter_config.json"))}

@app.post("/api/load_model")
async def load_model(base_model: str = Form(...), model_type: str = Form("base")):
    global chat_model, chat_tokenizer
    lora_path = f"saves/{base_model.split('/')[-1]}_lora_web"
    
    if not os.path.exists(base_model):
        return {"status": "error", "message": f"模型路径不存在: {base_model}"}

    try:
        print(f"📦 [1/3] 执行兼容补丁并加载配置: {base_model}")
        if not hasattr(nn.Module, "all_tied_weights_keys"):
            nn.Module.all_tied_weights_keys = {}

        config = AutoConfig.from_pretrained(base_model, trust_remote_code=True)
        print("🚀 [2/3] 正在将基座模型加载至 GPU...")
        chat_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_model, config=config, device_map="cuda:0", dtype=torch.float16, trust_remote_code=True
        )

        if model_type == "lora":
            if not os.path.exists(lora_path):
                return {"status": "error", "message": "未找到微调权重，请先完成微调！"}
            print(f"🔌 [3/3] 正在挂载 LoRA Adapter: {lora_path}")
            chat_model = PeftModel.from_pretrained(base, lora_path)
        else:
            chat_model = base
            
        chat_model.eval()
        print("✅ 模型已就绪！")
        return {"status": "success", "message": "加载成功！"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": f"加载失败: {str(e)}"}

@app.post("/api/chat")
async def chat_inference(prompt: str = Form(...)):
    global chat_model, chat_tokenizer
    if chat_model is None:
        return {"status": "error", "message": "请先在页面上方加载模型！"}
        
    if hasattr(chat_model.config, "max_length"):
        try: delattr(chat_model.config, "max_length")
        except Exception: pass
            
    if not hasattr(chat_model.config, "num_hidden_layers"):
        chat_model.config.num_hidden_layers = getattr(chat_model.config, "num_layers", 28)

    def _disable_dynamic_cache(*args, **kwargs): return False
    chat_model._supports_default_dynamic_cache = _disable_dynamic_cache
    if hasattr(chat_model, "base_model"): chat_model.base_model._supports_default_dynamic_cache = _disable_dynamic_cache

    def _extract_past_dummy(self, outputs, standardized_name="past_key_values"):
        return getattr(outputs, standardized_name, None)
        
    layers_to_patch = [chat_model]
    if hasattr(chat_model, "base_model"):
        layers_to_patch.append(chat_model.base_model)
        if hasattr(chat_model.base_model, "model"): layers_to_patch.append(chat_model.base_model.model)
            
    for layer in layers_to_patch:
        if not hasattr(layer, "_extract_past_from_model_output"):
            layer._extract_past_from_model_output = types.MethodType(_extract_past_dummy, layer)
            
    try:
        messages = [{"role": "user", "content": prompt}]
        text = chat_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = chat_tokenizer([text], return_tensors="pt").to(chat_model.device)
        
        with torch.no_grad():
            outputs = chat_model.generate(**inputs, max_new_tokens=512)
            outputs = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            response = chat_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
        return {"status": "success", "response": response}
    except Exception as e:
        return {"status": "error", "message": f"推理报错: {str(e)}"}

# ================= 评测模块 =================

def run_eval_background_task(base_model_path: str, lora_path: str, test_file: str):
    global chat_model, chat_tokenizer, eval_status_db
    try:
        print("\n🧹 [评测准备] 清空显存...")
        if 'chat_model' in globals() and chat_model is not None:
            del chat_model
            chat_model = None
            chat_tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        eval_status_db["status"] = "evaluating_base"
        rouge, smooth = Rouge(), SmoothingFunction().method1

        with open(test_file, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
        total_lines = len(lines)

        # ================= 🚨 致命修复：封装打补丁的函数 =================
        def apply_model_patches(m):
            if hasattr(m.config, "max_length"):
                try: delattr(m.config, "max_length")
                except Exception: pass
            if not hasattr(m.config, "num_hidden_layers"):
                m.config.num_hidden_layers = getattr(m.config, "num_layers", 28)
            def _disable_dynamic_cache(*args, **kwargs): return False
            m._supports_default_dynamic_cache = _disable_dynamic_cache
            if hasattr(m, "base_model"): m.base_model._supports_default_dynamic_cache = _disable_dynamic_cache
            def _extract_past_dummy(self, outputs, standardized_name="past_key_values"):
                return getattr(outputs, standardized_name, None)
            layers = [m]
            if hasattr(m, "base_model"):
                layers.append(m.base_model)
                if hasattr(m.base_model, "model"): layers.append(m.base_model.model)
            for layer in layers:
                if not hasattr(layer, "_extract_past_from_model_output"):
                    layer._extract_past_from_model_output = types.MethodType(_extract_past_dummy, layer)
        # =================================================================

        def evaluate_lines(model, tokenizer, desc):
            total_bleu, total_rouge_l, total_meteor = 0.0, 0.0, 0.0
            count = 0
            for idx, line in enumerate(lines):
                data = json.loads(line)
                instruction, reference = data["instruction"], data["output"]
                messages = [{"role": "user", "content": instruction}]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer([text], return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    # 添加 pad_token_id 防止模型生成时找不到终点无限死循环
                    outputs = model.generate(
                        **inputs, 
                        max_new_tokens=256,
                        pad_token_id=tokenizer.eos_token_id 
                    )
                    outputs = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
                    prediction = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                pred_tokens, ref_tokens = list(jieba.cut(prediction)), list(jieba.cut(reference))
                if pred_tokens and ref_tokens:
                    total_bleu += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
                    total_rouge_l += rouge.get_scores(" ".join(pred_tokens), " ".join(ref_tokens))[0]["rouge-l"]["f"]
                    total_meteor += meteor_score([ref_tokens], pred_tokens)
                count += 1
                
                base_progress = 0 if desc == "base" else 50
                eval_status_db["progress"] = base_progress + int((count / total_lines) * 50)
                print(f"📊 [{desc}] 评测进度: {count}/{total_lines}") # 让你在终端能看到心跳

            return {
                "BLEU": total_bleu / count if count else 0,
                "ROUGE-L": total_rouge_l / count if count else 0,
                "METEOR": total_meteor / count if count else 0,
            }

        print("🚀 [1/2] 评测原模型...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map="cuda:0", dtype=torch.float16, trust_remote_code=True
        )
        apply_model_patches(model) # 🚨 第一处打补丁
        model.eval()
        base_scores = evaluate_lines(model, tokenizer, "base")

        print(f"🔌 [2/2] 挂载并评测微调模型...")
        eval_status_db["status"] = "evaluating_lora"
        model = PeftModel.from_pretrained(model, lora_path)
        apply_model_patches(model) # 🚨 第二处打补丁
        model.eval()
        lora_scores = evaluate_lines(model, tokenizer, "lora")

        print("🧹 评测结束，释放显存...")
        del model
        gc.collect()
        torch.cuda.empty_cache()
        
        # 将结果存入全局变量供前端读取
        eval_status_db["status"] = "completed"
        eval_status_db["results"] = {"base": base_scores, "ft": lora_scores}

        # ================= 🚨 新增：物理存盘防丢失 =================
        with open("saves/latest_eval_results.json", "w", encoding="utf-8") as f:
            json.dump(eval_status_db["results"], f, ensure_ascii=False, indent=2)
        print("💾 成绩单已安全存入 saves/latest_eval_results.json")
        # ==============================================================

    except Exception as e:
        import traceback
        traceback.print_exc()
        eval_status_db["status"] = "error"
        eval_status_db["message"] = str(e)

@app.post("/api/start_eval")
async def start_eval(
    background_tasks: BackgroundTasks, 
    base_model: str = Form(...),
    file: UploadFile = File(...)  # 🚨 致命修复：接收真实的文件对象，而不是字符串！
):
    global eval_status_db
    
    # 1. 彻底告别时间戳文件夹扫描，直接指向确定的相对路径
    model_name = base_model.split('/')[-1] 
    lora_path = f"saves/{model_name}_lora_web"
    
    if not os.path.exists(lora_path) or not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        eval_status_db["status"] = "error"
        eval_status_db["message"] = f"找不到微调权重: {lora_path}，请确认是否已完成训练！"
        return {"status": "error", "message": eval_status_db["message"]}
        
    # ================= 🚨 新增：处理前端传来的文件 =================
    try:
        # 生成一个带时间戳的临时测试集名字
        test_file_path = f"data/test_set_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        content = await file.read()
        
        # 把前端传过来的测试数据保存到后端的 data 目录下
        with open(test_file_path, "wb") as f:
            f.write(content)
            
    except Exception as e:
        return {"status": "error", "message": f"读取测试集文件失败: {str(e)}"}
    # ==============================================================

    print(f"\n🎯 [路径扫描] 锁定微调权重: {lora_path}")
    print(f"📥 [数据接收] 已保存前端上传的测试集: {test_file_path}")
    
    # 把刚才保存的文件路径传给后台的评测狂魔
    background_tasks.add_task(run_eval_background_task, base_model, lora_path, test_file_path)
    
    eval_status_db["status"] = "starting"
    eval_status_db["progress"] = 0
    return {"status": "success", "message": "评测任务已在后台极速启动"}

@app.get("/api/eval_status")
async def get_eval_status():
    return eval_status_db

@app.post("/api/unload_model")
async def unload_model():
    """手动释放对话模型的显存"""
    global chat_model, chat_tokenizer
    try:
        # 1. 斩断 Python 变量的引用
        if chat_model is not None:
            del chat_model
            chat_model = None
        if chat_tokenizer is not None:
            del chat_tokenizer
            chat_tokenizer = None
            
        # 2. 强制 Python 进行垃圾回收
        gc.collect()
        # 3. 强制清空 CUDA 缓存，将显存真正还给操作系统
        torch.cuda.empty_cache()
        
        print("\n🧹 [显存管理] 对话模型已成功卸载，显存已释放。")
        return {"status": "success", "message": "显存释放成功！GPU 已清空。"}
    except Exception as e:
        return {"status": "error", "message": f"释放显存失败: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)