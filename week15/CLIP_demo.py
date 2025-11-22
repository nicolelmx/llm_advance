# ==============================================================================
# 多模态AI实战：CLIP模型教学演示
# ==============================================================================
# 本演示代码展示了CLIP模型的核心功能：
# 1. 零样本图像分类 - 展示CLIP的零样本迁移能力
# 2. 以文搜图 - 通过文本描述检索相关图片
# 3. 以图搜文 - 通过图片检索匹配的文本描述
# 4. 特征空间探索 - 展示CLIP如何将图像和文本映射到统一特征空间
#
# 【使用本地图片的方法】
# 如果在线图片URL无法访问，可以使用本地图片：
# 1. 将图片文件放在与脚本相同的目录下
# 2. 修改代码中的图片加载部分，例如：
#    image = Image.open('your_image.jpg')  # 替换URL下载部分
# 3. 或者设置 USE_LOCAL_IMAGES = True，然后指定本地图片路径
# ==============================================================================

# 步骤 1: 导入必要的库
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import numpy as np
import io
from torch.nn.functional import cosine_similarity

print("=" * 70)
print("多模态AI实战：CLIP模型核心功能演示")
print("=" * 70)
print("\n【CLIP模型简介】")
print("CLIP (Contrastive Language-Image Pre-training) 是OpenAI开发的多模态模型")
print("通过4亿图文对预训练，将图像和文本映射到统一的512维特征空间")
print("实现跨模态语义对齐，支持零样本迁移和跨模态检索\n")

# 步骤 2: 加载预训练的CLIP模型和处理器
# 模型是"大脑"，处理器是"助手"，负责将原始的图片和文字转换成模型能理解的格式。
# 我们使用OpenAI官方发布的经典模型 'openai/clip-vit-base-patch32'
MODEL_NAME = "openai/clip-vit-base-patch32"

# 【可选配置】使用本地图片
# 如果设置为True，将使用本地图片路径而不是在线URL
USE_LOCAL_IMAGES = False
LOCAL_IMAGE_PATHS = [
    "cat.jpg",      # 本地图片路径1
    "dog.jpg",      # 本地图片路径2
    "landscape.jpg", # 本地图片路径3
    "person.jpg"    # 本地图片路径4
]

print("【模型加载】")
print(f"正在加载模型: {MODEL_NAME}")
print("模型架构：ViT图像编码器 + Transformer文本编码器")
print("特征维度：512维统一嵌入空间")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("✓ 模型加载完成！\n")

# ==============================================================================
#  Part 1: 零样本图像分类 (Zero-Shot Image Classification)
# ==============================================================================
# 目标：给模型一张图片，让它从我们提供的文本标签中选出最匹配的一个。
# 原理：CLIP通过对比学习，将图像和文本映射到统一特征空间，计算相似度
# ==============================================================================
print("\n" + "=" * 70)
print("任务1: 零样本图像分类 (Zero-Shot Classification)")
print("=" * 70)
print("\n【核心原理】")
print("CLIP将分类任务转化为开放域图文匹配问题")
print("通过计算图像特征与文本标签特征的相似度，选择最匹配的标签")
print("无需针对特定类别进行微调，即可实现零样本分类\n")

# 步骤 3: 准备输入数据（一张图片和几个候选文本）
print("【数据准备】")
image = None

# 优先使用本地图片（如果配置了）
if USE_LOCAL_IMAGES and LOCAL_IMAGE_PATHS:
    for img_path in LOCAL_IMAGE_PATHS:
        try:
            print(f"尝试加载本地图片: {img_path}")
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = img
            print(f"✓ 本地图片加载成功！尺寸: {image.size}")
            break
        except FileNotFoundError:
            print(f"  ⚠ 文件不存在: {img_path}")
            continue
        except Exception as e:
            print(f"  ⚠ 加载失败: {str(e)[:50]}")
            continue

# 如果本地图片未配置或加载失败，尝试在线下载
if image is None:
    image_urls_to_try = [
        "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫的图片
        "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400",  # 狗的图片
        "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?w=400",  # 人物图片
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # 风景图片
    ]
    
    print("正在尝试下载在线图片...")
    for url in image_urls_to_try:
        try:
            print(f"尝试下载: {url[:60]}...")
            response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            
            img_data = io.BytesIO(response.content)
            img = Image.open(img_data)
            img.load()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            image = img
            print(f"✓ 图片下载成功！尺寸: {image.size}")
            break
        except Exception as e:
            print(f"  ⚠ 下载失败: {str(e)[:50]}")
            continue

# 如果所有方法都失败，创建备用图片
if image is None:
    print("\n⚠ 所有图片加载方式都失败")
    print("建议：")
    print("  1. 将图片保存到本地，设置 USE_LOCAL_IMAGES = True")
    print("  2. 修改 LOCAL_IMAGE_PATHS 为您的图片路径")
    print("  3. 或者使用：image = Image.open('your_image.jpg')")
    image = Image.new('RGB', (224, 224), color=(200, 200, 200))
    print("已创建一个备用灰色图片用于演示。")

# 准备我们的候选文本标签（注意：使用"a photo of"前缀可以提高CLIP的识别准确率）
text_labels = [
    "a photo of a cat", 
    "a photo of a dog", 
    "a photo of an astronaut riding a horse",
    "a photo of a bird",
    "a photo of a car"
]
print(f"\n候选文本标签 ({len(text_labels)}个):")
for i, label in enumerate(text_labels, 1):
    print(f"  {i}. {label}")

# 步骤 4: 数据预处理并进行推理
print("\n【特征提取与推理】")
print("1. 使用CLIP处理器将图片和文本转换为模型输入格式")
print("2. 图像编码器(ViT)提取图像特征向量")
print("3. 文本编码器(Transformer)提取文本特征向量")
print("4. 计算图像特征与各文本特征的相似度")

# 使用"助手"(processor)将图片和文本打包成模型需要的格式（PyTorch张量）
# padding=True 表示将所有文本处理成相同的长度
# return_tensors="pt" 表示返回PyTorch Tensors
inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)

# 将处理好的数据送入"大脑"(model)进行计算
# torch.no_grad() 表示我们只是在做推理，不需要计算梯度，这样可以节省计算资源
with torch.no_grad():
    outputs = model(**inputs)

# 模型的输出包含了各种信息，我们最关心的是图片与每个文本的相似度得分
# logits_per_image 是一个矩阵，表示每张图片与每个文本标签的原始匹配分数（logits）
logits_per_image = outputs.logits_per_image 

# 步骤 5: 解读并展示结果
# 为了让分数更直观，我们使用softmax函数将其转换为概率分布
# 这样所有标签的概率加起来会等于1
probs = logits_per_image.softmax(dim=1) 

print("\n【分类结果】")
print("-" * 70)
# 将概率和标签配对，并按概率从高到低排序
results = sorted(zip(text_labels, probs[0].tolist()), key=lambda x: x[1], reverse=True)

# 打印出每个标签及其对应的概率
for rank, (label, score) in enumerate(results, 1):
    bar_length = int(score * 50)  # 用条形图可视化概率
    bar = "█" * bar_length
    print(f"排名 {rank}: {label:<45} | 概率: {score:.4f} | {bar}")

best_label, best_score = results[0]
print("-" * 70)
print(f"\n✓ 最佳匹配: '{best_label}' (置信度: {best_score:.2%})")
print("\n【技术要点】")
print("• CLIP通过对比学习实现跨模态语义对齐")
print("• 零样本能力：无需针对特定类别进行训练")
print("• 统一特征空间：图像和文本映射到相同的512维空间")
print("=" * 70)


# ==============================================================================
#  Part 2: 以文搜图 (Text-to-Image Retrieval)
# ==============================================================================
# 目标：给定一个文本描述，从多张图片中找出最符合描述的那一张。
# 应用场景：智能相册搜索、内容审核、电商商品检索等
# ==============================================================================
print("\n" + "=" * 70)
print("任务2: 以文搜图 (Text-to-Image Retrieval)")
print("=" * 70)
print("\n【核心原理】")
print("1. 提取查询文本的特征向量")
print("2. 提取图库中所有图片的特征向量")
print("3. 计算文本特征与各图片特征的余弦相似度")
print("4. 按相似度排序，返回Top-K最相关图片\n")

# 步骤 1: 准备输入数据（一个文本描述和多张图片）
# 根据实际可用的图片调整查询文本
query_text = "a photo of a cat"
print("【数据准备】")
print(f"查询文本: '{query_text}'")
print("提示：如果图片无法加载，请使用本地图片或修改查询文本以匹配实际图片内容")

# 准备一组图片URL（使用更可靠的图片源）
# 如果这些URL无法访问，建议使用本地图片路径
image_urls = [
    "http://images.cocodataset.org/val2017/000000039769.jpg",  # 猫
    "https://images.unsplash.com/photo-1517849845537-4d257902454a?w=400",  # 狗
    "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400",  # 风景
    "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # 人物
]

# 为每张图片添加描述（用于后续展示）
image_descriptions = [
    "猫",
    "狗",
    "风景",
    "人物"
]

images = []
print(f"\n正在准备 {len(image_urls)} 张图片...")

# 优先使用本地图片（如果配置了）
if USE_LOCAL_IMAGES and LOCAL_IMAGE_PATHS:
    for i, img_path in enumerate(LOCAL_IMAGE_PATHS[:len(image_urls)]):
        try:
            print(f"  尝试加载本地图片 {i+1}: {img_path}")
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            images.append(img)
            print(f"  ✓ 图片 {i+1}: {image_descriptions[i]} - 尺寸: {img.size}")
        except FileNotFoundError:
            print(f"  ⚠ 文件不存在: {img_path}")
            colors = [(200, 150, 150), (150, 200, 150), (150, 150, 200), (200, 200, 150)]
            images.append(Image.new('RGB', (224, 224), color=colors[i % len(colors)]))
            print(f"     已创建备用图片")
        except Exception as e:
            print(f"  ⚠ 加载失败: {str(e)[:50]}")
            colors = [(200, 150, 150), (150, 200, 150), (150, 150, 200), (200, 200, 150)]
            images.append(Image.new('RGB', (224, 224), color=colors[i % len(colors)]))
            print(f"     已创建备用图片")

# 如果本地图片未配置或数量不足，尝试在线下载
if len(images) < len(image_urls):
    print("正在尝试下载在线图片...")
    for i, url in enumerate(image_urls):
        if i < len(images):
            continue  # 已经加载了本地图片
        try:
            # 添加User-Agent头，避免某些网站拒绝请求
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, timeout=15, headers=headers)
            response.raise_for_status()
            
            # 验证响应内容类型是否为图片
            content_type = response.headers.get('content-type', '')
            if content_type and not content_type.startswith('image/'):
                print(f"  ⚠ 警告：URL返回的Content-Type不是图片格式: {content_type}")
            
            # 将响应内容读取到内存中，然后使用BytesIO打开
            img_data = io.BytesIO(response.content)
            img = Image.open(img_data)
            img.load()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            images.append(img)
            print(f"  ✓ 图片 {i+1}: {image_descriptions[i]} - 尺寸: {img.size}")
        except (requests.exceptions.RequestException, ValueError, Image.UnidentifiedImageError, Exception) as e:
            print(f"  ⚠ 无法下载或识别图片 {i+1}")
            print(f"     URL: {url[:60]}...")
            print(f"     错误: {str(e)[:80]}")
            # 创建备用图片（不同颜色以便区分）
            colors = [(200, 150, 150), (150, 200, 150), (150, 150, 200), (200, 200, 150)]
            images.append(Image.new('RGB', (224, 224), color=colors[i % len(colors)]))
            print(f"     已创建备用图片（颜色标识）")

print(f"\n✓ 成功准备 {len(images)} 张图片用于检索\n")

# 步骤 2: 数据预处理并进行推理
print("【特征提取与相似度计算】")
print("正在计算文本特征与各图片特征的相似度...")

# 这次我们有多张图片和一个文本
inputs = processor(text=[query_text], images=images, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# 这次我们关心的是每个图片与这个文本的匹配度
# logits_per_text 是一个矩阵，表示每个文本与每张图片的原始匹配分数
logits_per_text = outputs.logits_per_text

# 同样，使用softmax转换为概率
probs = logits_per_text.softmax(dim=1)

# 步骤 3: 解读并展示结果
print("\n【检索结果】")
print("-" * 70)
# 将概率和图片信息配对，并按概率从高到低排序
results = sorted(zip(range(len(images)), image_descriptions, image_urls, probs[0].tolist()), 
                 key=lambda x: x[3], reverse=True)

for rank, (idx, desc, url, score) in enumerate(results, 1):
    bar_length = int(score * 50)
    bar = "█" * bar_length
    print(f"排名 {rank}: {desc:<15} | 相似度: {score:.4f} | {bar}")
    print(f"         URL: {url[:60]}...")

print("-" * 70)
best_idx, best_desc, best_url, best_score = results[0]
print(f"\n✓ 最佳匹配: 图片 '{best_desc}' (相似度: {best_score:.2%})")
print("\n【技术要点】")
print("• 余弦相似度：衡量文本和图像特征向量的方向一致性")
print("• 批量检索：可同时处理多张图片，提高检索效率")
print("• 实际应用：召回率可达85%以上（Flickr30K数据集）")
print("=" * 70)


# ==============================================================================
#  Part 3: 以图搜文 (Image-to-Text Retrieval)
# ==============================================================================
# 目标：给定一张图片，从多个文本描述中找出最匹配的那一个。
# 应用场景：自动图像标注、内容理解、智能推荐等
# ==============================================================================
print("\n" + "=" * 70)
print("任务3: 以图搜文 (Image-to-Text Retrieval)")
print("=" * 70)
print("\n【核心原理】")
print("1. 提取查询图片的特征向量")
print("2. 提取文本库中所有文本的特征向量")
print("3. 计算图片特征与各文本特征的余弦相似度")
print("4. 按相似度排序，返回Top-K最相关文本描述\n")

# 步骤 1: 准备输入数据（一张图片和多个文本描述）
print("【数据准备】")
# 使用之前下载的图片（例如第一张）
query_image = images[0] if images else Image.new('RGB', (224, 224), color='white')
print(f"查询图片: 尺寸 {query_image.size}")

# 准备多个候选文本描述
candidate_texts = [
    "a photo of an astronaut riding a horse",
    "a photo of a cat sitting on a sofa",
    "a photo of a dog playing in the park",
    "a photo of a red car parked on the street",
    "a photo of a beautiful landscape with mountains",
    "a photo of multiple cats lying together",
    "a photo of a person reading a book",
    "a photo of a bicycle on the road"
]

print(f"\n候选文本描述 ({len(candidate_texts)}个):")
for i, text in enumerate(candidate_texts, 1):
    print(f"  {i}. {text}")

# 步骤 2: 数据预处理并进行推理
print("\n【特征提取与相似度计算】")
print("正在计算图片特征与各文本特征的相似度...")

inputs = processor(text=candidate_texts, images=query_image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# 使用 logits_per_image 表示图片与各文本的匹配度
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)

# 步骤 3: 解读并展示结果
print("\n【检索结果】")
print("-" * 70)
results = sorted(zip(candidate_texts, probs[0].tolist()), key=lambda x: x[1], reverse=True)

for rank, (text, score) in enumerate(results, 1):
    bar_length = int(score * 50)
    bar = "█" * bar_length
    print(f"排名 {rank}: {text:<55} | 相似度: {score:.4f} | {bar}")

print("-" * 70)
best_text, best_score = results[0]
print(f"\n✓ 最佳匹配: '{best_text}' (相似度: {best_score:.2%})")
print("\n【技术要点】")
print("• 双向检索：CLIP支持文本→图像和图像→文本的双向检索")
print("• 语义理解：能够理解图像的高级语义，而非仅依赖低级特征")
print("• 实际应用：可用于自动图像标注、内容审核等场景")
print("=" * 70)


# ==============================================================================
#  Part 4: 特征空间探索 (Feature Space Exploration)
# ==============================================================================
# 目标：展示CLIP如何将图像和文本映射到统一特征空间
# 通过可视化特征向量，理解跨模态语义对齐机制
# ==============================================================================
print("\n" + "=" * 70)
print("任务4: 特征空间探索 (Feature Space Exploration)")
print("=" * 70)
print("\n【核心原理】")
print("CLIP通过对比学习，将图像和文本映射到统一的512维特征空间")
print("语义相似的图像和文本在特征空间中距离更近")
print("通过分析特征向量的相似度，可以理解模型的跨模态理解能力\n")

# 步骤 1: 提取多个图像和文本的特征向量
print("【特征提取】")
sample_images = images[:3] if len(images) >= 3 else images
sample_texts = [
    "a photo of a cat",
    "a photo of a dog", 
    "a photo of sightseeing"
]

# 准备图像描述（用于显示）
sample_image_descriptions = image_descriptions[:len(sample_images)] if len(image_descriptions) >= len(sample_images) else [f"图片{i+1}" for i in range(len(sample_images))]

print(f"提取 {len(sample_images)} 张图片和 {len(sample_texts)} 个文本的特征向量...")

# 分别提取图像特征和文本特征
with torch.no_grad():
    # 提取图像特征
    image_inputs = processor(images=sample_images, return_tensors="pt")
    image_features = model.get_image_features(**image_inputs)
    # 归一化特征向量（CLIP的标准做法）
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # 提取文本特征
    text_inputs = processor(text=sample_texts, return_tensors="pt", padding=True)
    text_features = model.get_text_features(**text_inputs)
    # 归一化特征向量
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

print(f"✓ 图像特征维度: {image_features.shape}")  # [num_images, 512]
print(f"✓ 文本特征维度: {text_features.shape}")    # [num_texts, 512]

# 步骤 2: 计算跨模态相似度矩阵
print("\n【相似度矩阵分析】")
print("计算图像特征与文本特征的相似度矩阵...")

# 计算相似度矩阵：image_features @ text_features.T
similarity_matrix = (image_features @ text_features.T) * 100  # 乘以100是为了与logits scale一致
similarity_matrix = similarity_matrix.cpu().numpy()

print("\n相似度矩阵 (图像 × 文本):")
print("-" * 70)
print(f"{'图像/文本':<20}", end="")
for text in sample_texts:
    print(f"{text[:20]:<22}", end="")
print()

for i, img_desc in enumerate(sample_image_descriptions):
    print(f"{img_desc:<20}", end="")
    for j in range(len(sample_texts)):
        print(f"{similarity_matrix[i][j]:>20.2f}", end="  ")
    print()

print("-" * 70)
print("\n【解读】")
print("• 数值越大表示相似度越高（范围通常在-100到100之间）")
print("• 对角线或接近对角线的值通常较高，表示匹配的图文对")
print("• 通过对比学习，CLIP学会了将语义相关的图像和文本映射到相近的位置")

# 步骤 3: 展示特征向量的统计信息
print("\n【特征向量统计】")
print("分析特征向量的分布特性...")

print(f"\n图像特征统计:")
print(f"  • 均值: {image_features.mean().item():.4f}")
print(f"  • 标准差: {image_features.std().item():.4f}")
print(f"  • 特征向量范数: {image_features.norm(dim=-1).mean().item():.4f} (归一化后应为1.0)")

print(f"\n文本特征统计:")
print(f"  • 均值: {text_features.mean().item():.4f}")
print(f"  • 标准差: {text_features.std().item():.4f}")
print(f"  • 特征向量范数: {text_features.norm(dim=-1).mean().item():.4f} (归一化后应为1.0)")

print("\n【技术要点】")
print("• 统一特征空间：图像和文本共享512维嵌入空间")
print("• 归一化处理：特征向量经过L2归一化，便于计算余弦相似度")
print("• 对比学习：通过最大化匹配对的相似度，最小化不匹配对的相似度来训练")
print("=" * 70)


# ==============================================================================
#  课程总结
# ==============================================================================
print("\n" + "=" * 70)
print("课程总结")
print("=" * 70)
print("\n【CLIP核心能力回顾】")
print("1. ✓ 零样本图像分类：无需微调即可识别新类别")
print("2. ✓ 以文搜图：通过文本描述检索相关图片")
print("3. ✓ 以图搜文：通过图片检索匹配的文本描述")
print("4. ✓ 跨模态理解：将不同文本映射到统一特征空间")

print("\n【关键技术要点】")
print("• 对比学习：通过4亿图文对预训练，实现跨模态语义对齐")
print("• 双编码器架构：ViT图像编码器 + Transformer文本编码器")
print("• 统一嵌入空间：512维特征空间，支持跨模态检索")
print("• 零样本迁移：预训练后可直接应用于下游任务，无需微调")

print("\n【实际应用场景】")
print("• 智能相册：通过文字搜索照片")
print("• 内容审核：图文匹配检测")
print("• 电商检索：商品图片与描述匹配")
print("• 自动标注：为图片生成文字描述")

print("\n【扩展学习方向】")
print("• LLaVA：多模态大模型，支持视觉对话")
print("• Stable Diffusion：文生图技术")
print("• 向量数据库：FAISS/ChromaDB用于大规模检索")
print("• 多模态Agent：结合LangChain构建智能体")

print("\n" + "=" * 70)
print("感谢学习！多模态AI的世界充满无限可能！")
print("=" * 70)

