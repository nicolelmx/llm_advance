# 任务：请根据 `week5_assignment_instructions.md` 中的说明，修改此文件以完成作业。
# 你需要实现以下三个核心任务：
# 1. 将模型中的 LSTM 层替换为 GRU 层。
# 2. 将原始的 LSTM 层修改为双向 (Bidirectional) LSTM。
# 3. 为模型添加 Dropout 正则化层。

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

# 1. 数据准备 (无需修改)
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

# 2. 定义模型 (*** 在这里开始修改 ***)
# -----------------------------------------------------------------------------
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(SentimentClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # --- 任务一 & 任务二 在这里修改 ---
        # 原始的 LSTM 层
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 任务一: 请将上面的LSTM层注释掉，并在这里实现一个 GRU 层
        # self.gru = nn.GRU(...) input_size, hidden_size,batch_first
        
        # ---------------------------------
        
        
        # --- 任务三 在这里修改 ---
        # 在这里定义一个 Dropout 层
        self.dropout = nn.Dropout(0.5)
        
        # ---------------------------------
        
        
        # --- 任务二 可能需要修改这里 ---
        # 原始的全连接层
        self.fc = nn.Linear(hidden_dim, output_dim)
        # ---------------------------------

    def forward(self, text):
        embedded = self.embedding(text)
        
        # --- 任务一 在这里修改 ---
        # 使用 LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用 GRU (需要取消注释并做相应修改)
        # gru_out, hidden = self.gru(embedded)
        
        # ---------------------------------
        
        # 根据你使用的是单向还是双向模型，hidden的形状会有所不同。
        # 对于双向模型，你可能需要拼接前向和后向的隐藏状态。
        # 这里我们简化处理，只取最后一个时间步的隐藏状态。
        # 对于单向模型, hidden 的形状是 [1, batch_size, hidden_dim]
        # 对于双向模型, hidden 的形状是 [2, batch_size, hidden_dim]
        
        final_hidden_state = hidden.squeeze(0) # 简化处理，只取了正向的最后一个hidden state

        # --- 任务三 在这里修改 ---
        # 在这里应用 Dropout 层
        
        # ---------------------------------
        
        output = self.fc(final_hidden_state)
        return torch.sigmoid(output)

# 3. 训练模型 (大部分无需修改)
# -----------------------------------------------------------------------------
EMBEDDING_DIM = 16
HIDDEN_DIM = 32
OUTPUT_DIM = 1
LEARNING_RATE = 0.05
EPOCHS = 300

# 根据你的实现实例化模型
model = SentimentClassifier(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCELoss()

print("开始训练模型...")
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

# 4. 测试模型 (无需修改)
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
