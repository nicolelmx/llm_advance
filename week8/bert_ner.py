import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
import inspect
from datasets import Dataset
import numpy as np
import seqeval.metrics
import warnings
warnings.filterwarnings('ignore')

# --- 模块二：与时俱进 - Fine-tuning BERT 实现更高性能NER ---

# --- 第一部分：拥抱Transformers ---

# 为了与bilstm_crf_ner.py脚本保持一致，我们使用相同的数据
# 但需要将其转换为Hugging Face `datasets`库所期望的格式
raw_data = {
    "tokens": [
        "马 云 创 办 了 阿 里 巴 巴".split(),
        "李 彦 宏 是 百 度 的 创 始 人".split(),
        "我 爱 北 京 天 安 门".split(),
    ],
    "tags": [
        "B-PER I-PER O O O B-ORG I-ORG I-ORG I-ORG".split(),
        "B-PER I-PER I-PER O B-ORG I-ORG O O O O".split(),
        "O O B-LOC I-LOC B-LOC I-LOC I-LOC".split(),
    ],
}

# 创建标签到ID的映射
unique_tags = set(tag for doc in raw_data["tags"] for tag in doc)
tag2id = {tag: i for i, tag in enumerate(unique_tags)}
id2tag = {i: tag for tag, i in tag2id.items()}
# 将标签转换为ID
raw_data["ner_tags"] = [[tag2id[tag] for tag in doc] for doc in raw_data["tags"]]

# 转换为Dataset对象
dataset = Dataset.from_dict(raw_data)

# 1. Hugging Face Transformers 库入门
# 加载预训练模型的tokenizer
model_checkpoint = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


# 2. BERT 特有的数据处理
def tokenize_and_align_labels(examples):
    # examples["tokens"] 是一个句子列表, is_split_into_words=True 表示它已经被分词
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # 特殊token（如[CLS], [SEP]）的word_id为None，我们将其标签设为-100
            if word_idx is None:
                label_ids.append(-100)
            # 如果是词的第一个token，使用其原始标签
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # 如果是词的后续token，也设为-100（或跟随第一个token的标签，取决于策略）
            else:
                label_ids.append(-100) # 忽略后续token的损失
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 将处理函数应用到整个数据集
tokenized_dataset = dataset.map(tokenize_and_align_labels, batched=True)


# --- 第二部分：模型微调与王者对决 ---

# 1. 模型微调 (Fine-tuning)
# 加载预训练模型，并指定标签数量
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint, num_labels=len(unique_tags), id2label=id2tag, label2id=tag2id
)

# 定义训练参数（通过签名动态适配不同Transformers版本）
sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

kwargs = {
    "output_dir": "temp_ner_model",
    "learning_rate": 2e-5,
    "num_train_epochs": 10,
    "weight_decay": 0.01,
    "logging_steps": 1,
    "report_to": "none"
}

# 评估策略/开关
if "evaluation_strategy" in sig_params:
    kwargs["evaluation_strategy"] = "epoch"
elif "do_eval" in sig_params:
    kwargs["do_eval"] = True

# 批大小参数兼容（per_device vs per_gpu）
if "per_device_train_batch_size" in sig_params:
    kwargs["per_device_train_batch_size"] = 2
elif "per_gpu_train_batch_size" in sig_params:
    kwargs["per_gpu_train_batch_size"] = 2

if "per_device_eval_batch_size" in sig_params:
    kwargs["per_device_eval_batch_size"] = 2
elif "per_gpu_eval_batch_size" in sig_params:
    kwargs["per_gpu_eval_batch_size"] = 2

args = TrainingArguments(**kwargs)

# 数据整理器，用于将样本动态填充到批次中的最大长度
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

# 定义评估指标计算函数
def compute_metrics(p):
    # 兼容不同版本的Transformers：有的传 EvalPrediction 对象，有的传 (predictions, labels) 元组
    if hasattr(p, "predictions"):
        predictions = p.predictions
        labels = p.label_ids
    else:
        predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # 移除-100的标签
    true_predictions = [
        [id2tag[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2tag[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 使用seqeval计算指标
    results = seqeval.metrics.classification_report(true_labels, true_predictions, output_dict=True)
    
    return {
        "precision": results["weighted avg"]["precision"],
        "recall": results["weighted avg"]["recall"],
        "f1": results["weighted avg"]["f1-score"],
        "accuracy": seqeval.metrics.accuracy_score(true_labels, true_predictions),
    }


# 实例化Trainer
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,  # 简单起见，用训练集做评估
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
print("\n--- 开始微调 BERT 模型 ---")
trainer.train()
print("--- 训练完成 ---")


# 2. 结果对比与深度分析
print("\n--- 模型预测示例 ---")
from transformers import pipeline

# 使用pipeline进行推理
text = "马 云 在 阿 里 巴 巴 工 作"
ner_pipe = pipeline("ner", model=model, tokenizer=tokenizer)
results = ner_pipe(text)

print(f"测试句子: {text}")
print("预测结果:")
for entity in results:
    print(entity)

# 优势分析:
# 1. 强大的语义表示: BERT在大规模语料上预训练，语义理解能力远超从零训练的Embedding。
# 2. 上下文理解: Transformer的自注意力机制能捕捉更长距离的依赖关系。
# 3. OOV问题缓解: WordPiece分词机制能很好地处理未登录词。
#
# 劣势与权衡:
# 1. 速度慢: 模型大，计算密集，推理速度远慢于BiLSTM。
# 2. 资源消耗大: 需要GPU进行高效训练，模型文件也很大。
# 工业界常在效果和成本间做权衡，可能会选择蒸馏后的小模型或其他轻量化方案。
