import json
import numpy as np
import matplotlib.pyplot as plt
import os

# 1. 设置中文字体，确保中文正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei'] 
plt.rcParams['axes.unicode_minus'] = False

# 2. 读取保存的成绩单
json_path = 'saves/latest_eval_results.json'
if not os.path.exists(json_path):
    print(f"❌ 找不到成绩单文件：{json_path}")
    exit(1)

with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 3. 🚨 剔除 EM，只保留三大核心自然语言生成指标
metrics = ['BLEU', 'ROUGE-L', 'METEOR']

base_scores = [data['base'].get(m, 0) for m in metrics]
ft_scores = [data['ft'].get(m, 0) for m in metrics]

# 4. 动态计算天花板（最大值的 1.15 倍，给顶点留点呼吸空间）
max_scores = [max(b, f) * 1.15 if max(b, f) > 0 else 0.1 for b, f in zip(base_scores, ft_scores)]

# 归一化数据（映射到 0~1 的比例，画图用）
base_norm = [b / m for b, m in zip(base_scores, max_scores)]
ft_norm = [f / m for f, m in zip(ft_scores, max_scores)]

# 闭合多边形（把第一个点复制到最后，构成封闭三角形）
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
base_norm += base_norm[:1]
ft_norm += ft_norm[:1]
angles += angles[:1]

# 5. 开始画图
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

# 让第一个指标 (BLEU) 稳稳地站在正上方
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

# 画雷达网格线和外围标签
plt.xticks(angles[:-1], metrics, color='#333333', size=14, fontweight='bold')
ax.set_yticklabels([])  # 隐藏内部圈圈的丑陋数字
ax.grid(color='#E5E5E5', linestyle='-', linewidth=1.5)
ax.spines['polar'].set_color('#CCCCCC')

# ================= 绘制 原模型 (Base) =================
# 原模型用深沉的蓝色，低调一点
ax.plot(angles, base_norm, color='#4A90E2', linewidth=2, linestyle='solid', label='Base Model')
ax.fill(angles, base_norm, color='#4A90E2', alpha=0.25)

# 给原模型打上真实的数值标签
for i, angle in enumerate(angles[:-1]):
    # 稍微往圆心收一点，避免和外圈文字撞车
    ax.text(angle, base_norm[i] - 0.12, f"{base_scores[i]:.4f}", color='#357ABD', size=10, ha='center', va='center')

# ================= 绘制 微调模型 (Fine-tuned) =================
# 微调模型用亮眼的翠绿色，粗线条，强烈包裹
ax.plot(angles, ft_norm, color='#50E3C2', linewidth=3.5, linestyle='solid', label='Fine-tuned')
ax.fill(angles, ft_norm, color='#50E3C2', alpha=0.45)

# 给微调模型打上真实的数值标签
for i, angle in enumerate(angles[:-1]):
    # 稍微往外扩一点
    ax.text(angle, ft_norm[i] + 0.12, f"{ft_scores[i]:.4f}", color='#008F66', size=12, fontweight='bold', ha='center', va='center')

# 6. 图例与标题
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=False)
plt.title("Performance Comparison", size=20, fontweight='bold', y=1.12)

# 7. 保存高清大图
output_path = 'saves/radar_chart_3D_hd.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
print(f"🎉 帅气的三维性能雷达图生成成功！已保存至：{output_path}")