import json
import torch
import jieba
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import jieba
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import os
class ModelEvaluator:
    def __init__(self, base_model_path, lora_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        self.model_name = "Base Model (原模型)"
        if lora_path:
            print(f"\n正在挂载 LoRA 权重: {lora_path} ...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            self.model_name = "Fine-tuned Model (微调后)"
        self.model.eval()

    def generate_answer(self, instruction):
        messages = [{"role": "user", "content": instruction}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=256)
            outputs = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
            return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    def evaluate(self, test_file):
        print(f"\n开始评估: {self.model_name}")
        rouge, smooth = Rouge(), SmoothingFunction().method1
        total_bleu, total_rouge_l, total_meteor, total_em, count = 0.0, 0.0, 0.0, 0.0, 0
        
        with open(test_file, 'r', encoding='utf-8') as f:
            lines = [line for line in f if line.strip()]
            
        for line in tqdm(lines, desc="推理打分中"):
            data = json.loads(line)
            instruction, reference = data["instruction"], data["output"]
            
            prediction = self.generate_answer(instruction)
            pred_tokens, ref_tokens = list(jieba.cut(prediction)), list(jieba.cut(reference))
            
            if not pred_tokens or not ref_tokens: continue
            
            # 1. BLEU (相似度)
            total_bleu += sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smooth)
            # 2. ROUGE-L (召回率)
            total_rouge_l += rouge.get_scores(" ".join(pred_tokens), " ".join(ref_tokens))[0]["rouge-l"]["f"]
            # 3. METEOR (综合语义匹配)
            total_meteor += meteor_score([ref_tokens], pred_tokens)
            # 4. EM (完全匹配 Exact Match) - 去除首尾空白后严苛比对
            if prediction.strip() == reference.strip():
                total_em += 1.0
                
            count += 1
            
        return {
            "BLEU": total_bleu / count, 
            "ROUGE-L": total_rouge_l / count,
            "METEOR": total_meteor / count,
            "EM": total_em / count
        } if count > 0 else {"BLEU": 0, "ROUGE-L": 0, "METEOR": 0, "EM": 0}

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # 替换为你的真实基座路径
    LORA_PATH = "saves/Qwen2.5-7B-Instruct_lora_web" # 你的微调输出路径
    TEST_FILE = "data/test_data.jsonl"       # 你的测试集路径
    
    # 1. 评估原模型
    evaluator = ModelEvaluator(BASE_MODEL)
    base_scores = evaluator.evaluate(TEST_FILE)
    
    # 清理显存
    del evaluator
    torch.cuda.empty_cache()
    
    # 2. 评估微调模型
    evaluator_ft = ModelEvaluator(BASE_MODEL, LORA_PATH)
    ft_scores = evaluator_ft.evaluate(TEST_FILE)
    
    # 3. 打印终极对比表
    print("\n" + "="*40)
    print(f"{'指标':<12} | {'原模型':<12} | {'微调后':<12} | {'提升'}")
    print("-" * 40)
    print(f"{'BLEU':<10} | {base_scores['BLEU']:.4f}       | {ft_scores['BLEU']:.4f}       | +{(ft_scores['BLEU'] - base_scores['BLEU']):.4f}")
    print(f"{'ROUGE-L':<10} | {base_scores['ROUGE-L']:.4f}       | {ft_scores['ROUGE-L']:.4f}       | +{(ft_scores['ROUGE-L'] - base_scores['ROUGE-L']):.4f}")
    print("="*40)