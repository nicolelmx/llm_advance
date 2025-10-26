import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from peft import get_peft_model, LoraConfig, TaskType
# 使用seqeval进行实体识别任务的评估
import evaluate

# --- 环境设置 ---
# 确保Hugging Face Hub的顺利访问
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 设置WANDB为离线模式，避免需要登录
os.environ["WANDB_DISABLED"] = "true"

# --- 1. 数据集加载与预处理 ---
try:
    # 尝试使用CoNLL-2003英文NER数据集作为替代
    raw_datasets = load_dataset("conll2003")
    # 为了快速演示，我们只使用一小部分数据
    train_dataset = raw_datasets["train"].select(range(1000))
    validation_dataset = raw_datasets["validation"].select(range(200))
    print("CoNLL-2003 NER数据集加载成功！")
    print("训练集样本数:", len(train_dataset))
    print("验证集样本数:", len(validation_dataset))
except Exception as e:
    print(f"CoNLL-2003数据集加载失败: {e}")
    print("创建模拟NER数据集...")
    
    # 创建模拟的中文NER数据集
    from datasets import Dataset
    
    # 模拟中文NER数据
    mock_data = {
        "tokens": [
            ["张", "三", "在", "北", "京", "大", "学", "工", "作"],
            ["李", "四", "来", "自", "上", "海", "市"],
            ["王", "五", "是", "清", "华", "大", "学", "的", "学", "生"],
            ["赵", "六", "在", "深", "圳", "腾", "讯", "公", "司", "上", "班"],
            ["钱", "七", "住", "在", "广", "州", "市", "天", "河", "区"]
        ] * 200,  # 重复200次创建1000个样本
        "ner_tags": [
            [1, 2, 0, 3, 4, 5, 6, 0, 0],  # 张三(PER), 北京大学(ORG)
            [1, 2, 0, 0, 3, 4, 5, 0],     # 李四(PER), 上海市(LOC)
            [1, 2, 0, 3, 4, 5, 6, 0, 0, 0], # 王五(PER), 清华大学(ORG)
            [1, 2, 0, 3, 4, 5, 6, 7, 8, 0, 0], # 赵六(PER), 深圳腾讯公司(ORG)
            [1, 2, 0, 0, 3, 4, 5, 6, 7, 8]  # 钱七(PER), 广州市天河区(LOC)
        ] * 200
    }
    
    # 创建训练和验证数据集
    train_dataset = Dataset.from_dict({
        "tokens": mock_data["tokens"][:800],
        "ner_tags": mock_data["ner_tags"][:800]
    })
    validation_dataset = Dataset.from_dict({
        "tokens": mock_data["tokens"][800:1000],
        "ner_tags": mock_data["ner_tags"][800:1000]
    })
    
    print("模拟中文NER数据集创建成功！")
    print("训练集样本数:", len(train_dataset))
    print("验证集样本数:", len(validation_dataset))

# 获取标签列表
try:
    # 尝试从数据集特征中获取标签列表（适用于HuggingFace数据集）
    label_list = train_dataset.features["ner_tags"].feature.names
    print("实体标签列表:", label_list)
except AttributeError:
    # 如果是模拟数据集，手动定义标签列表
    label_list = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG", "B-MISC", "I-MISC"]
    print("使用预定义的实体标签列表:", label_list)
    
    # 为模拟数据集添加特征信息
    from datasets import Features, Sequence, Value, ClassLabel
    features = Features({
        'tokens': Sequence(Value('string')),
        'ner_tags': Sequence(ClassLabel(names=label_list))
    })
    
    # 重新创建数据集以包含正确的特征
    train_dataset = train_dataset.cast(features)
    validation_dataset = validation_dataset.cast(features)

# --- 2. 分词与标签对齐 ---
model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_and_align_labels(examples):
    """处理单个样本，进行分词和标签对齐"""
    # 检查输入格式，适配不同的数据集结构
    if "tokens" in examples:
        # 模拟数据集格式：已经分词的tokens
        texts = examples["tokens"]
    elif "text" in examples:
        # 原始文本格式
        texts = examples["text"]
    else:
        raise ValueError("数据集必须包含 'tokens' 或 'text' 字段")
    
    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        is_split_into_words=True,  # 输入是已经分好词的列表
        padding=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS] or [SEP] token
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # 新的词元
                if word_idx < len(label):
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
            else:
                # 同一个词元的其他子词元
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 对数据集进行处理
tokenized_train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
tokenized_validation_dataset = validation_dataset.map(tokenize_and_align_labels, batched=True)

# 创建数据整理器
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# --- 3. 评估指标定义 ---
seqeval = evaluate.load("seqeval")

def compute_metrics(p):
    """计算评估指标"""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 将索引转换回标签字符串
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# --- 4. LoRA 配置与模型加载 ---

# 加载基础的BERT模型
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    num_labels=len(label_list),
    id2label={str(i): label for i, label in enumerate(label_list)},
    label2id={label: str(i) for i, label in enumerate(label_list)},
)

# 打印模型结构，观察可训练参数
print("\n--- 原始BERT模型 ---")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"可训练参数比例: {100 * trainable_params / total_params:.2f}%")

# 定义LoRA配置
lora_config = LoraConfig(
    task_type=TaskType.TOKEN_CLS, # 任务类型为Token Classification
    inference_mode=False,
    r=8,                          # 低秩矩阵的秩
    lora_alpha=16,                # LoRA的alpha参数
    lora_dropout=0.1,             # Dropout率
    bias="none",                  # 不训练偏置项
    # 通常在注意力机制的query和value层应用LoRA
    target_modules=["query", "value"],
)

# 使用get_peft_model将基础模型转换为PEFT模型
model = get_peft_model(model, lora_config)

# 打印转换后模型的结构，观察可训练参数量的巨大变化
print("\n--- LoRA PEFT 模型 ---")
model.print_trainable_parameters()


# --- 5. 训练参数设置与Trainer初始化 ---

# 定义训练参数
training_args = TrainingArguments(
    output_dir="lora-bert-ner-clue",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",  # 修复：使用 eval_strategy 而不是 evaluation_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
    logging_steps=50,
    fp16=torch.cuda.is_available(), # 如果有GPU，开启半精度训练
    report_to=None,  # 禁用wandb等报告工具
)

# 初始化Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_validation_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# --- 6. 模型训练 ---
print("\n--- 开始LoRA微调 ---")
try:
    trainer.train()
    print("\n--- 训练完成 ---")
except Exception as e:
    print(f"\n训练过程中发生错误: {e}")
    print("请检查GPU显存是否充足。如果显存不足，可以尝试减小 'per_device_train_batch_size'。")
    exit()

# --- 7. 保存LoRA适配器 ---
output_path = "lora-bert-ner-adapter"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)
print(f"\nLoRA适配器已保存到: {output_path}")
print("注意：这里只保存了轻量的适配器文件，而不是整个BERT模型。")


# --- 8. 使用微调后的模型进行推理 ---
print("\n--- 使用微调后的LoRA模型进行推理 ---")
text = "爱奇艺的《庆余年第二季》在北京开机，主演包括张若昀和李沁。"
print(f"测试文本: {text}")

# 加载我们保存的适配器进行推理
try:
    # 重新加载基础模型
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        num_labels=len(label_list)
    )
    # 加载PEFT模型并附加适配器权重
    from peft import PeftModel
    peft_model = PeftModel.from_pretrained(base_model, output_path)
    
    # 使用pipeline进行推理
    from transformers import pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=peft_model,
        tokenizer=tokenizer,
        aggregation_strategy="simple" # 将同一个实体的子词元合并
    )
    
    results = ner_pipeline(text)
    print("\n命名实体识别结果:")
    for entity in results:
        print(f"  实体: {entity['word']}, 类型: {entity['entity_group']}, 得分: {entity['score']:.4f}")

except Exception as e:
    print(f"\n推理失败: {e}")
    print("你可以尝试手动运行推理代码，或检查保存的模型路径。")
