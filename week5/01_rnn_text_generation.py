import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

# 1. 数据准备 (升级)
# -----------------------------------------------------------------------------
corpus = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
""".lower()

chars = sorted(list(set(corpus)))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}
vocab_size = len(chars)

# 将整个语料库切分为固定长度的序列进行训练
seq_length = 40
input_seqs = []
target_seqs = []
for i in range(len(corpus) - seq_length):
    input_seqs.append([char_to_idx[c] for c in corpus[i:i+seq_length]])
    target_seqs.append([char_to_idx[c] for c in corpus[i+1:i+seq_length+1]])

# 2. 定义RNN模型
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        # (新) 词嵌入层，将字符索引转换为密集向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # RNN层
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        # 全连接层，将隐藏状态映射回词汇表
        self.fc = nn.Linear(hidden_size, vocab_size)

    # 前向传播方法：定义输入如何变成输出
    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        # 初始化隐藏状态
        return torch.zeros(1, batch_size, self.hidden_size)

# 3. 训练模型
# -----------------------------------------------------------------------------
# 定义模型参数
embedding_dim = 16
hidden_size = 64
learning_rate = 0.005
epochs = 500

model = CharRNN(vocab_size, embedding_dim, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("开始训练升级版RNN模型...")
for epoch in range(epochs):
    # (新) 每次随机选择一个序列进行训练，这是一种简单的随机梯度下降
    seq_idx = random.randint(0, len(input_seqs) - 1)
    
    input_tensor = torch.tensor(input_seqs[seq_idx]).unsqueeze(0) # 增加batch维度 -> [1, seq_length]
    target_tensor = torch.tensor(target_seqs[seq_idx]) # [seq_length]

    hidden = model.init_hidden()
    optimizer.zero_grad()

    outputs, hidden = model(input_tensor, hidden) # outputs: [1, seq_length, vocab_size]
    
    # 调整输出和目标的形状以匹配损失函数的要求
    loss = criterion(outputs.squeeze(0), target_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print("训练完成！")

# 4. 生成文本 (温度采样)
# -----------------------------------------------------------------------------
def generate_text(model, start_char, length, temperature=0.8):
    """
    生成文本的函数
    temperature: 控制随机性的参数。
        - 值越高(>1.0)，生成的文本越随机、越有“创意”；
        - 值越低(<1.0)，生成的文本越保守、越接近模型学到的模式。
        - 值为1.0时，按原始概率分布采样。
    """
    model.eval() # 切换到评估模式
    with torch.no_grad():
        result = start_char
        input_char = torch.tensor([char_to_idx[start_char]]).unsqueeze(0)
        hidden = model.init_hidden()

        for _ in range(length):
            output, hidden = model(input_char, hidden)
            
            # (新) 温度采样逻辑
            output_dist = output.squeeze().div(temperature).exp()
            # 从给定的概率分布中随机抽取指定数量的样本
            top_i = torch.multinomial(output_dist, 1)[0]
            
            predicted_char = idx_to_char[top_i.item()]
            result += predicted_char
            
            # 将当前预测的字符作为下一个时间步的输入
            input_char = torch.tensor([top_i.item()]).unsqueeze(0)
            
    return result

# 尝试不同的温度来观察生成效果
print("\n--- 生成文本 (温度: 0.5 - 比较保守) ---")
print(generate_text(model, 't', 200, temperature=0.5))

print("\n--- 生成文本 (温度: 1.0 - 更有创意) ---")
print(generate_text(model, 't', 200, temperature=1.0))

print("\n--- 生成文本 (温度: 1.5 - 可能开始胡言乱语) ---")
print(generate_text(model, 't', 200, temperature=1.5))
