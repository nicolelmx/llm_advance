# week5_assignment_solution.py
# 这是 week5_assignment_student.py 的参考答案版本。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# 1. 数据准备 (与学生版文件相同)
# -----------------------------------------------------------------------------
train_data = [
    ("this movie is great", 1),
    ("i love this film", 1),
    ("what a fantastic show", 1),
    ("the plot is boring", 0),
    ("i did not like the acting", 0),
    ("it was a waste of time", 0),
    ("the storyline was predictable", 0),
    ("a truly heartwarming story", 1),
]

word_to_idx = {"<PAD>": 0}
for sentence, _ in train_data:
    for word in sentence.split():
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
vocab_size = len(word_to_idx)
idx_to_word = {i: w for w, i in word_to_idx.items()}

sequences = [torch.tensor([word_to_idx[w] for w in s.split()]) for s, _ in train_data]
labels = torch.tensor([label for _, label in train_data], dtype=torch.float32)

padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=word_to_idx["<PAD>"])

# 2. 定义模型 (参考答案实现)
# -----------------------------------------------------------------------------
# 参考答案整合了所有任务要求：
# 1. 使用 GRU 层 (任务一)
# 2. 设置为双向 (任务二)
# 3. 添加 Dropout (任务三)

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # 任务一 & 二: 定义一个双向 GRU 层
        self.gru = nn.GRU(embedding_dim, 
                          hidden_dim, 
                          batch_first=True, 
                          bidirectional=True)
        
        # 任务三: 定义 Dropout 层
        self.dropout = nn.Dropout(dropout_rate)
        
        # 任务二的后续修改: 全连接层的输入维度需要加倍，因为它接收来自两个方向的隐藏状态
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        # 通过 GRU 层
        # gru_out shape: [batch_size, seq_len, hidden_dim * 2]
        # hidden shape: [num_layers * 2, batch_size, hidden_dim] -> [2, batch_size, hidden_dim]
        gru_out, hidden = self.gru(embedded)
        
        # 拼接前向 (hidden[0]) 和后向 (hidden[1]) 的最后一个隐藏状态
        final_hidden_state = torch.cat((hidden[0, :, :], hidden[1, :, :]), dim=1)
        # final_hidden_state shape: [batch_size, hidden_dim * 2]
        
        # 应用 Dropout
        dropped_out = self.dropout(final_hidden_state)
        # dropped_out shape: [batch_size, hidden_dim * 2]
        
        # 通过全连接层
        output = self.fc(dropped_out)
        # output shape: [batch_size, output_dim]
        
        return torch.sigmoid(output)

# 3. 训练模型 (与学生版文件相同)
# -----------------------------------------------------------------------------
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 1
LEARNING_RATE = 0.05
EPOCHS = 300

model = SentimentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print("开始训练模型 (参考答案版)...")
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(padded_sequences).squeeze(1)
    loss = criterion(predictions, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 30 == 0:
        rounded_preds = torch.round(predictions)
        correct = (rounded_preds == labels).float()
        accuracy = correct.sum() / len(correct)
        print(f'Epoch: {epoch+1:03}, Loss: {loss.item():.4f}, Accuracy: {accuracy.item()*100:.2f}%')

print("训练完成！")

# 4. 测试模型 (与学生版文件相同)
# -----------------------------------------------------------------------------
def predict_sentiment(model, sentence):
    model.eval()
    with torch.no_grad():
        words = sentence.split()
        indexed = [word_to_idx.get(w, 0) for w in words]
        tensor = torch.LongTensor(indexed).unsqueeze(0)
        prediction = model(tensor)
        return "正面" if prediction.item() > 0.5 else "负面"

test_sentence_1 = "this film is fantastic"
print(f"'{test_sentence_1}' 的情感是: {predict_sentiment(model, test_sentence_1)}")

test_sentence_2 = "the storyline was terrible"
print(f"'{test_sentence_2}' 的情感是: {predict_sentiment(model, test_sentence_2)}")

test_sentence_3 = "i absolutely did not enjoy this movie"
print(f"'{test_sentence_3}' 的情感是: {predict_sentiment(model, test_sentence_3)}")
