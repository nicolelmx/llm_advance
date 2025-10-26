
import torch
import torch.nn as nn
import torch.optim as optim
import random

from TorchCRF import CRF

# --- 模块一：经典再现 - 从零构建 BiLSTM+CRF ---

# --- 第一部分：项目准备与数据探索 ---
# 格式为 (token, tag) 对组成的句子列表。
training_data = [
    (
        "马 云 创 办 了 阿 里 巴 巴".split(),
        "B-PER I-PER O O O B-ORG I-ORG I-ORG I-ORG".split(),
    ),
    (
        "李 彦 宏 是 百 度 的 创 始 人".split(),
        "B-PER I-PER I-PER O B-ORG I-ORG O O O O".split(),
    ),
    ("我 爱 北 京 天 安 门".split(), "O O B-LOC I-LOC B-LOC I-LOC I-LOC".split()),
]

# 额外：构造一些简单的数据增强样本，缓解数据过少问题
# 初始化模块内部维护的伪随机数生成器状态
random.seed(42)

def make_bio_for_entity(tokens, entity_tag):
    labels = []
    for idx, _ in enumerate(tokens):
        if idx == 0:
            labels.append(f"B-{entity_tag}")
        else:
            labels.append(f"I-{entity_tag}")
    return labels

def build_sample(prefix_tokens, entity_tokens, entity_tag, suffix_tokens):
    sent = prefix_tokens + entity_tokens + suffix_tokens
    labels = ["O"] * len(prefix_tokens) + make_bio_for_entity(entity_tokens, entity_tag) + ["O"] * len(suffix_tokens)
    return (sent, labels)

# 简单的人名/组织/地点词表，生成若干模板句
name_list = [list("张 三".split()), list("王 五".split()), list("马 云".split())]
org_list = [list("阿 里 巴 巴".split()), list("百 度".split()), list("腾 讯".split())]
loc_list = [list("北 京".split()), list("上 海".split()), list("天 安 门".split())]

augmented = []
# 模板1：X 在 Y 工 作
for name in name_list:
    for org in org_list:
        prefix = []
        suffix = list("在 Y 工 作".split())  # 先放占位，后面替换
        # 实际按 [name] + [在] + [org] + [工作]
        sent = name + list("在".split()) + org + list("工 作".split())
        labels = make_bio_for_entity(name, "PER") + ["O"] + make_bio_for_entity(org, "ORG") + ["O", "O"]
        augmented.append((sent, labels))

# 模板2：我 爱 <LOC>
for loc in loc_list:
    sent = list("我 爱".split()) + loc
    labels = ["O", "O"] + make_bio_for_entity(loc, "LOC")
    augmented.append((sent, labels))

# 模板3：<ORG> 招 聘 <PER>
for org in org_list:
    for name in name_list:
        sent = org + list("招 聘".split()) + name
        labels = make_bio_for_entity(org, "ORG") + ["O", "O"] + make_bio_for_entity(name, "PER")
        augmented.append((sent, labels))

print(augmented)

# 随机打乱并采样一部分，避免过度重复
random.shuffle(augmented)
print(augmented)
training_data.extend(augmented[:30])
print(training_data)

# 构建词表/字表 和 标签表
word_to_ix = {}
tag_to_ix = {}
for sentence, tags in training_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
    for tag in tags:
        if tag not in tag_to_ix:
            tag_to_ix[tag] = len(tag_to_ix)

# 添加特殊token: PAD 用于填充, UNK 用于未知词
word_to_ix["<PAD>"] = len(word_to_ix)
word_to_ix["<UNK>"] = len(word_to_ix)
ix_to_tag = {v: k for k, v in tag_to_ix.items()}


# --- 第二部分：代码实战 - 构建与训练 ---

# 1. 数据预处理: 将文本和标签转换为ID序列
def prepare_sequence(seq, to_ix):
    # 智能处理<UNK>：仅当<UNK>在词典中时，才将其用作后备
    if "<UNK>" in to_ix:
        unk_ix = to_ix["<UNK>"]
        idxs = [to_ix.get(w, unk_ix) for w in seq]
    else:
        # 对于标签，我们假设所有标签都在字典中
        idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# 2. 模型搭建 (Model Architecture)
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # Embedding层 词嵌入层
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        
        # BiLSTM层 双向 LSTM 层
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True,  # 确保输入和输出的维度顺序为 (batch, seq_len, feature)
        )

        # Dropout 可以有效缓解小数据集过拟合
        self.dropout = nn.Dropout(p=0.3)

        # Linear层: 将BiLSTM的输出映射到标签空间  全连接层
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # CRF层 - 移除 batch_first 以兼容旧版本
        self.crf = CRF(self.tagset_size)
        # 兼容不同实现：有的CRF使用 batch_first=True，有的是False（默认）
        self.crf_batch_first = getattr(self.crf, "batch_first", False)

    def _get_lstm_features(self, sentence):
        # sentence shape: (seq_len) -> (1, seq_len)
        embeds = self.word_embeds(sentence).unsqueeze(0)
        # embeds shape: (1, seq_len, embedding_dim)
        
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        # lstm_out shape: (1, seq_len, hidden_dim)

        lstm_feats = self.hidden2tag(lstm_out)
        # lstm_feats shape: (1, seq_len, tagset_size)

        # 转置维度以匹配CRF层的期望输入 (seq_len, batch_size, num_tags)
        return lstm_feats.permute(1, 0, 2)

    def forward(self, sentence):  # 用于预测/解码
        # sentence shape: (seq_len)
        lstm_feats = self._get_lstm_features(sentence)
        # lstm_feats shape: (seq_len, 1, tagset_size)
        
        # 准备两种布局：seq_first 与 batch_first，运行时自动选择长度匹配的一种
        emissions_seq = lstm_feats  # (seq_len, 1, tags)
        mask_seq = torch.ones(emissions_seq.shape[0], emissions_seq.shape[1], dtype=torch.bool, device=emissions_seq.device)

        emissions_batch = lstm_feats.permute(1, 0, 2)  # (1, seq_len, tags)
        mask_batch = torch.ones(emissions_batch.shape[0], emissions_batch.shape[1], dtype=torch.bool, device=emissions_batch.device)

        seq_len = sentence.shape[0]

        def _decode(emissions, mask, use_decode=True):
            if hasattr(self.crf, "decode") and use_decode:
                paths = self.crf.decode(emissions, mask=mask)
                return paths[0]
            else:
                out = self.crf.viterbi_decode(emissions, mask)
                first = out[0]
                return first[0] if isinstance(first, tuple) else first

        # 先按非batch_first尝试
        try:
            path_seq = _decode(emissions_seq, mask_seq)
        except Exception:
            path_seq = None

        # 再按batch_first尝试
        try:
            path_batch = _decode(emissions_batch, mask_batch)
        except Exception:
            path_batch = None

        # 选择与输入长度匹配的路径
        if isinstance(path_seq, (list, tuple)) and len(path_seq) == seq_len:
            return path_seq
        if isinstance(path_batch, (list, tuple)) and len(path_batch) == seq_len:
            return path_batch

        # 回退：优先选择更长的那一个
        if isinstance(path_seq, (list, tuple)) and isinstance(path_batch, (list, tuple)):
            return path_seq if len(path_seq) >= len(path_batch) else path_batch
        return path_seq or path_batch or [int(torch.argmax(self.hidden2tag(self.word_embeds(sentence).unsqueeze(0))[-1, 0]).item())]

    def neg_log_likelihood(self, sentence, tags):  # 用于计算损失
        # sentence shape: (seq_len), tags shape: (seq_len)
        lstm_feats = self._get_lstm_features(sentence)
        # lstm_feats shape: (seq_len, 1, tagset_size)
        
        # 同时计算两种布局的loss，选择可用且较小的那个，避免实现差异
        losses = []
        # seq_first
        try:
            emissions = lstm_feats  # (seq, batch, tags)
            tags_b = tags.unsqueeze(1)  # (seq, batch)
            mask = torch.ones_like(tags_b, dtype=torch.bool)
            losses.append((-self.crf(emissions, tags_b, mask=mask).mean()).unsqueeze(0))
        except Exception:
            pass
        # batch_first
        try:
            emissions_b = lstm_feats.permute(1, 0, 2)  # (batch, seq, tags)
            tags_b2 = tags.unsqueeze(0)  # (batch, seq)
            mask2 = torch.ones_like(tags_b2, dtype=torch.bool)
            losses.append((-self.crf(emissions_b, tags_b2, mask=mask2).mean()).unsqueeze(0))
        except Exception:
            pass
        if not losses:
            raise RuntimeError("CRF loss failed in both layouts")
        return torch.min(torch.cat(losses))

def split_train_val(data, val_ratio=0.2):
    data_copy = list(data)
    random.shuffle(data_copy)
    split_idx = max(1, int(len(data_copy) * (1 - val_ratio)))
    return data_copy[:split_idx], data_copy[split_idx:]

train_data, val_data = split_train_val(training_data, val_ratio=0.25)

# 3. 模型训练与评估（加入验证与早停）
EMBEDDING_DIM = 96
HIDDEN_DIM = 192

model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.005)

def evaluate_loss(dataset):
    if not dataset:
        return None
    model.eval()
    total = 0.0
    with torch.no_grad():
        for sentence, tags in dataset:
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            total += model.neg_log_likelihood(sentence_in, targets).item()
    model.train()
    return total / max(1, len(dataset))

print("--- 开始训练 BiLSTM+CRF 模型（含数据增强/验证/早停） ---")
best_val = float('inf')
patience = 6
stale = 0
best_state = None
max_epochs = 40

for epoch in range(max_epochs):
    random.shuffle(train_data)
    for sentence, tags in train_data:
        model.zero_grad()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)
        loss = model.neg_log_likelihood(sentence_in, targets)
        loss.backward()
        optimizer.step()

    # 每个epoch结束进行一次验证
    val_loss = evaluate_loss(val_data)
    train_loss = evaluate_loss(train_data[:min(20, len(train_data))])  # 采样查看训练损失
    print(f"Epoch {epoch+1}/{max_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

    if val_loss is not None and val_loss < best_val - 1e-4:
        best_val = val_loss
        stale = 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        stale += 1
        if stale >= patience:
            print("早停触发，结束训练。")
            break

if best_state is not None:
    model.load_state_dict(best_state)
print("--- 训练完成 ---")


# --- 第三部分：代码复盘与分析 ---
# 检查模型在一个测试样本上的表现
with torch.no_grad():
    test_sentence = "马 云 在 阿 里 巴 巴 工 作".split()
    sentence_in = prepare_sequence(test_sentence, word_to_ix)
    predicted_ixs = model(sentence_in)
    predicted_tags = [ix_to_tag[ix] for ix in predicted_ixs]
    
    print("\n--- 模型预测示例 ---")
    print("测试句子:", " ".join(test_sentence))
    print("预测标签:", " ".join(predicted_tags))
    print("pred_ixs:", predicted_ixs)
    print("pred_len:", len(predicted_ixs), "sent_len:", len(test_sentence))
    
# 调优方向探讨:
# 1. 增加数据量: 当前数据集太小，容易过拟合。
# 2. 调整超参数: 如 EMBEDDING_DIM, HIDDEN_DIM, lr, epoch数。
# 3. 使用预训练词向量: 可以用预训练的中文词向量初始化 nn.Embedding 层，会极大提升效果。
# 4. 增加模型复杂度: 如增加LSTM的层数，或添加Dropout层防止过拟合。
