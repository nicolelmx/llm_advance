# ==============================================================================
# CLIP模型实战作业 - 学员版
# ==============================================================================
# 作业要求：
# 1. 完成以下TODO标记的部分
# 2. 运行代码验证结果
# 3. 理解CLIP模型的工作原理
# ==============================================================================

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import numpy as np
import io

print("=" * 70)
print("CLIP模型实战作业")
print("=" * 70)

# ==============================================================================
# 任务1: 模型加载
# ==============================================================================
print("\n【任务1: 模型加载】")
print("请完成模型和处理器的加载")

# TODO 1.1: 定义模型名称（使用 'openai/clip-vit-base-patch32'）
MODEL_NAME = None  # 请在此处填写模型名称

# TODO 1.2: 加载CLIP模型
# 提示：使用 CLIPModel.from_pretrained()
model = None  # 请在此处加载模型

# TODO 1.3: 加载CLIP处理器
# 提示：使用 CLIPProcessor.from_pretrained()
processor = None  # 请在此处加载处理器

print("✓ 模型加载完成！\n")

# ==============================================================================
# 任务2: 零样本图像分类
# ==============================================================================
print("=" * 70)
print("【任务2: 零样本图像分类】")
print("=" * 70)
print("目标：给定一张图片，从候选标签中选出最匹配的一个\n")

# 准备测试图片
image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
print(f"正在下载测试图片: {image_url}")

try:
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    img_data = io.BytesIO(response.content)
    image = Image.open(img_data)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    print(f"✓ 图片下载成功！尺寸: {image.size}\n")
except Exception as e:
    print(f"⚠ 图片下载失败: {e}")
    image = Image.new('RGB', (224, 224), color='white')
    print("已创建备用图片\n")

# 候选文本标签
text_labels = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of an astronaut riding a horse",
    "a photo of a bird",
    "a photo of a car"
]

print(f"候选文本标签 ({len(text_labels)}个):")
for i, label in enumerate(text_labels, 1):
    print(f"  {i}. {label}")

# TODO 2.1: 使用processor处理图片和文本
# 提示：processor(text=..., images=..., return_tensors="pt", padding=True)
inputs = None  # 请在此处处理输入数据

# TODO 2.2: 使用模型进行推理（记得使用 torch.no_grad()）
# 提示：outputs = model(**inputs)
with torch.no_grad():
    outputs = None  # 请在此处进行推理

# TODO 2.3: 获取图像与文本的相似度分数
# 提示：使用 outputs.logits_per_image
logits_per_image = None  # 请在此处获取logits

# TODO 2.4: 将logits转换为概率分布
# 提示：使用 softmax 函数，dim=1
probs = None  # 请在此处计算概率

# 显示结果
print("\n【分类结果】")
print("-" * 70)
results = sorted(zip(text_labels, probs[0].tolist()), key=lambda x: x[1], reverse=True)

for rank, (label, score) in enumerate(results, 1):
    bar_length = int(score * 50)
    bar = "█" * bar_length
    print(f"排名 {rank}: {label:<45} | 概率: {score:.4f} | {bar}")

best_label, best_score = results[0]
print("-" * 70)
print(f"\n✓ 最佳匹配: '{best_label}' (置信度: {best_score:.2%})")

# ==============================================================================
# 任务3: 以文搜图
# ==============================================================================
print("\n" + "=" * 70)
print("【任务3: 以文搜图】")
print("=" * 70)
print("目标：给定文本描述，从多张图片中找出最匹配的一张\n")

# 查询文本
query_text = "a photo of a cat"
print(f"查询文本: '{query_text}'")

# 准备图片URL列表
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",
    "http://images.cocodataset.org/val2017/000000039623.jpg",
    "http://images.cocodataset.org/val2017/000000039631.jpg",
]

image_descriptions = ["图片1", "图片2", "图片3"]

# 下载图片
images = []
print(f"\n正在下载 {len(image_urls)} 张图片...")
for i, url in enumerate(image_urls):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img_data = io.BytesIO(response.content)
        img = Image.open(img_data)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)
        print(f"  ✓ 图片 {i+1} 下载成功")
    except Exception as e:
        print(f"  ⚠ 图片 {i+1} 下载失败: {e}")
        images.append(Image.new('RGB', (224, 224), color=(100, 100, 100)))

print(f"\n✓ 成功准备 {len(images)} 张图片\n")

# TODO 3.1: 处理查询文本和多张图片
# 提示：processor(text=[query_text], images=images, ...)
inputs = None  # 请在此处处理输入

# TODO 3.2: 进行推理
with torch.no_grad():
    outputs = None  # 请在此处进行推理

# TODO 3.3: 获取文本与各图片的相似度分数
# 提示：使用 outputs.logits_per_text
logits_per_text = None  # 请在此处获取logits

# TODO 3.4: 转换为概率分布
probs = None  # 请在此处计算概率

# 显示结果
print("【检索结果】")
print("-" * 70)
results = sorted(zip(range(len(images)), image_descriptions, probs[0].tolist()), 
                 key=lambda x: x[2], reverse=True)

for rank, (idx, desc, score) in enumerate(results, 1):
    bar_length = int(score * 50)
    bar = "█" * bar_length
    print(f"排名 {rank}: {desc:<15} | 相似度: {score:.4f} | {bar}")

best_idx, best_desc, best_score = results[0]
print("-" * 70)
print(f"\n✓ 最佳匹配: {best_desc} (相似度: {best_score:.2%})")

# ==============================================================================
# 任务4: 特征提取和相似度计算
# ==============================================================================
print("\n" + "=" * 70)
print("【任务4: 特征提取和相似度计算】")
print("=" * 70)
print("目标：提取图像和文本特征，计算它们之间的相似度\n")

# 使用第一张图片和几个文本
query_image = images[0] if images else Image.new('RGB', (224, 224), color='white')
sample_texts = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
]

print(f"查询图片: 尺寸 {query_image.size}")
print(f"候选文本: {sample_texts}\n")

# TODO 4.1: 提取图像特征
# 提示：使用 model.get_image_features()
# 步骤：
#   1. 使用 processor 处理图像: processor(images=[query_image], return_tensors="pt")
#   2. 使用 model.get_image_features(**image_inputs) 提取特征
#   3. 归一化特征向量: features / features.norm(dim=-1, keepdim=True)
with torch.no_grad():
    image_inputs = None  # 请在此处处理图像输入
    image_features = None  # 请在此处提取图像特征
    image_features = None  # 请在此处归一化特征向量

# TODO 4.2: 提取文本特征
# 提示：使用 model.get_text_features()
# 步骤：
#   1. 使用 processor 处理文本: processor(text=sample_texts, return_tensors="pt", padding=True)
#   2. 使用 model.get_text_features(**text_inputs) 提取特征
#   3. 归一化特征向量
with torch.no_grad():
    text_inputs = None  # 请在此处处理文本输入
    text_features = None  # 请在此处提取文本特征
    text_features = None  # 请在此处归一化特征向量

# TODO 4.3: 计算相似度矩阵
# 提示：相似度 = image_features @ text_features.T
# 注意：归一化后的特征向量，点积就是余弦相似度
similarity_matrix = None  # 请在此处计算相似度矩阵

# 显示结果
print("【相似度矩阵】")
print("-" * 70)
print(f"{'图像/文本':<20}", end="")
for text in sample_texts:
    print(f"{text[:20]:<22}", end="")
print()

similarity_np = similarity_matrix.cpu().numpy()
print(f"{'查询图片':<20}", end="")
for j in range(len(sample_texts)):
    print(f"{similarity_np[0][j]:>20.2f}", end="  ")
print()

print("-" * 70)
print("\n【解读】")
print("• 数值越大表示相似度越高")
print("• 相似度范围通常在 -1 到 1 之间（归一化后的余弦相似度）")
print("• 值接近1表示非常相似，接近-1表示非常不相似")

# ==============================================================================
# 作业总结
# ==============================================================================
print("\n" + "=" * 70)
print("作业完成！")
print("=" * 70)
print("\n【学习要点回顾】")
print("1. CLIP模型加载：使用 from_pretrained() 方法")
print("2. 数据预处理：使用 processor 处理图片和文本")
print("3. 模型推理：使用 model() 或 model.get_*_features() 方法")
print("4. 相似度计算：通过 logits 或特征向量的点积计算")
print("5. 概率转换：使用 softmax 将 logits 转换为概率分布")
print("\n【思考题】")
print("1. 为什么CLIP能够实现零样本分类？")
print("2. 图像特征和文本特征为什么可以计算相似度？")
print("3. 归一化特征向量有什么作用？")
print("=" * 70)

