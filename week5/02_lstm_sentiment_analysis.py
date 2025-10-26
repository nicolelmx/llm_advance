# 更复杂，更能体现LSTM处理长距离依赖的能力。
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# 1. 数据准备
# 训练数据：(评论, 标签)，0代表负面，1代表正面
train_data = [
    ("this movie is great", 1),
    ("i love this film", 1),
    ("what a fantastic show", 1),
    ("the plot is boring", 0),
    ("i did not like the acting", 0),
    ("it was a waste of time", 0),
]

# 构建词汇表
word_to_idx = {"<PAD>": 0}  # <PAD> 是用于填充的特殊标记
for sentence, _ in train_data:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
vocab_size = len(word_to_idx)
idx_to_word = {i: w for w, i in word_to_idx.items()}

# 将句子转换为索引序列
sequences = [torch.tensor([word_to_idx[w] for w in s.split()]) for s, _ in train_data]
labels = torch.tensor([label for _, label in train_data], dtype=torch.float32)

# 填充序列，使它们长度一致
# 传统的神经网络要求输入大小固定，因此我们需要将不同长度的句子填充到相同的长度。
# 这对应了PPT中提到的传统模型处理序列数据的挑战之一。
padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=word_to_idx["<PAD>"])


# 2. 定义LSTM模型
# 相比基础RNN，LSTM通过精巧的门控机制（遗忘门、输入门、输出门）来解决梯度消失问题，
# 从而能更好地捕捉句子中的长距离依赖关系。
class LSTMSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMSentimentClassifier, self).__init__()

        # 1. 词嵌入层 (Embedding Layer)
        # 将每个单词的索引映射到一个密集的词向量。
        # 这是比One-Hot更高效的表示方法。
        # num_embeddings 词汇表大小
        # embedding_dim 嵌入维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # 2. LSTM层
        # 接收词向量序列作为输入，并输出隐藏状态。
        # batch_first=True 表示输入的第一个维度是batch_size。
        # hidden_dim LSTM 隐藏层的维度
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # 3. 全连接层 (分类器)
        # 我们只取用LSTM最后一个时间步的隐藏状态，因为它被认为是整个句子的语义摘要。
        # 这对应PPT中提到的，在文本分类任务中，我们通常只关心 h_T。
        # in_features（必选）输入特征的维度（即输入向量的长度）。
        # out_features（必选）输出特征的维度（即输出向量的长度）。
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text):
        # text: [batch_size, seq_len]

        embedded = self.embedding(text)
        # embedded: [batch_size, seq_len, embedding_dim]

        # LSTM的输出包括所有时间步的输出和最后一个时间步的隐藏状态(h_n)与细胞状态(c_n)
        # 我们这里只需要最后一个隐藏状态 h_n 来代表整个句子。
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # hidden: [num_layers, batch_size, hidden_dim]
        # 我们只需要最后一个隐藏状态，所以取 hidden.squeeze(0)
        final_hidden_state = hidden.squeeze(0)

        # 通过全连接层和Sigmoid函数得到最终的概率
        output = self.fc(final_hidden_state)
        return torch.sigmoid(output)


# 3. 训练模型
# 定义模型参数
EMBEDDING_DIM = 10
HIDDEN_DIM = 32
OUTPUT_DIM = 1
LEARNING_RATE = 0.1
EPOCHS = 200

model = LSTMSentimentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()  # 二元交叉熵损失，适用于二分类问题

print("开始训练LSTM情感分类模型...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    predictions = model(padded_sequences).squeeze(1)

    # 计算损失
    loss = criterion(predictions, labels)

    # 反向传播
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        # 计算准确率
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == labels).float()
        accuracy = correct.sum() / len(correct)
        print(f'Epoch: {epoch + 1:02}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item() * 100:.2f}%')

print("训练完成！")


# 4. 测试模型 (推理)
def predict_sentiment(model, sentence):
    model.eval()
    with torch.no_grad():
        # 将句子转换为索引序列
        words = sentence.split()
        indexed = [word_to_idx.get(w, 0) for w in words]  # 如果词不在词汇表中，用<PAD>索引
        tensor = torch.LongTensor(indexed).unsqueeze(0)  # 增加batch维度

        # 预测
        prediction = model(tensor)

        return "正面" if prediction.item() > 0.5 else "负面"


# 测试新句子
test_sentence_1 = "this film is fantastic"
print(f"'{test_sentence_1}' 的情感是: {predict_sentiment(model, test_sentence_1)}")

test_sentence_2 = "the acting was terrible"
print(f"'{test_sentence_2}' 的情感是: {predict_sentiment(model, test_sentence_2)}")
