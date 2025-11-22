# ==============================================================================
# CLIP模型实战作业 - 参考答案（教师版）
# ==============================================================================
# 本文件包含完整的实现代码，供教师参考和学员对照学习
# ==============================================================================

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import numpy as np
import io

print("=" * 70)
print("CLIP模型实战作业 - 参考答案")
print("=" * 70)

# ==============================================================================
# 任务1: 模型加载
# ==============================================================================
print("\n【任务1: 模型加载】")
print("请完成模型和处理器的加载")

# 参考答案 1.1: 定义模型名称
MODEL_NAME = "openai/clip-vit-base-patch32"

# 参考答案 1.2: 加载CLIP模型
model = CLIPModel.from_pretrained(MODEL_NAME)

# 参考答案 1.3: 加载CLIP处理器
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

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

# 参考答案 2.1: 使用processor处理图片和文本
inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)

# 参考答案 2.2: 使用模型进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 参考答案 2.3: 获取图像与文本的相似度分数
logits_per_image = outputs.logits_per_image

# 参考答案 2.4: 将logits转换为概率分布
probs = logits_per_image.softmax(dim=1)

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

# 参考答案 3.1: 处理查询文本和多张图片
inputs = processor(text=[query_text], images=images, return_tensors="pt", padding=True)

# 参考答案 3.2: 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 参考答案 3.3: 获取文本与各图片的相似度分数
logits_per_text = outputs.logits_per_text

# 参考答案 3.4: 转换为概率分布
probs = logits_per_text.softmax(dim=1)

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

# 参考答案 4.1: 提取图像特征
with torch.no_grad():
    # 处理图像输入
    image_inputs = processor(images=[query_image], return_tensors="pt")
    # 提取图像特征
    image_features = model.get_image_features(**image_inputs)
    # 归一化特征向量（L2归一化）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

# 参考答案 4.2: 提取文本特征
with torch.no_grad():
    # 处理文本输入
    text_inputs = processor(text=sample_texts, return_tensors="pt", padding=True)
    # 提取文本特征
    text_features = model.get_text_features(**text_inputs)
    # 归一化特征向量（L2归一化）
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

# 参考答案 4.3: 计算相似度矩阵
# 归一化后的特征向量，点积就是余弦相似度
similarity_matrix = image_features @ text_features.T

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

# 额外说明
print("\n【技术细节说明】")
print("1. 特征提取：")
print("   - get_image_features() 提取图像特征向量（512维）")
print("   - get_text_features() 提取文本特征向量（512维）")
print("2. 归一化：")
print("   - L2归一化：features / features.norm(dim=-1, keepdim=True)")
print("   - 归一化后，特征向量的模长为1")
print("3. 相似度计算：")
print("   - 归一化向量的点积 = 余弦相似度")
print("   - 范围：[-1, 1]，1表示完全相同，-1表示完全相反")

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
print("\n【思考题答案提示】")
print("1. 为什么CLIP能够实现零样本分类？")
print("   答：CLIP通过对比学习，在4亿图文对上预训练，学会了将图像和文本")
print("      映射到统一的特征空间。因此，即使遇到新的类别，只要用文本描述，")
print("      就能通过特征相似度匹配，无需针对该类别进行训练。")
print("\n2. 图像特征和文本特征为什么可以计算相似度？")
print("   答：CLIP将图像和文本都映射到同一个512维的特征空间，在这个空间中，")
print("      语义相似的图像和文本会被映射到相近的位置。因此可以通过计算")
print("      特征向量的相似度（如余弦相似度）来衡量语义相似性。")
print("\n3. 归一化特征向量有什么作用？")
print("   答：归一化后，特征向量的模长变为1，此时两个向量的点积就等于")
print("      它们的余弦相似度。这样可以：")
print("      - 消除向量长度的影响，只关注方向（语义）")
print("      - 相似度范围固定在[-1, 1]，便于理解和比较")
print("      - 提高数值稳定性")
print("=" * 70)

