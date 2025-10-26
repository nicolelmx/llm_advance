# 循环神经网络作业 - 改进版（带反向传播）
# 简单序列分类：判断句子情感

import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置jieba日志级别
jieba.setLogLevel(20)

class SimpleRNN:
    def __init__(self, vocab_size, hidden_size, output_size, learning_rate=0.1):
        """简单RNN网络"""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重（使用更好的初始化方法）
        self.Wxh = np.random.randn(vocab_size, hidden_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(hidden_size, output_size) * 0.1
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        """Tanh导数"""
        return 1 - np.tanh(x) ** 2
    
    def softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向传播"""
        seq_len, batch_size, _ = X.shape
        self.h = np.zeros((seq_len + 1, batch_size, self.hidden_size))
        self.y = np.zeros((seq_len, batch_size, self.output_size))
        self.x = X  # 保存输入用于反向传播
        
        for t in range(seq_len):
            # 隐藏层: h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            self.h[t+1] = self.tanh(
                np.dot(X[t], self.Wxh) + 
                np.dot(self.h[t], self.Whh) + 
                self.bh
            )
            # 输出层: y_t = softmax(Why * h_t + by)
            self.y[t] = self.softmax(
                np.dot(self.h[t+1], self.Why) + 
                self.by
            )
        
        return self.y
    
    def backward(self, y_true):
        """反向传播"""
        seq_len, batch_size, _ = self.x.shape
        
        # 初始化梯度
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        
        # 计算最后一个时间步的损失梯度
        dy = self.y[-1] - y_true  # 交叉熵损失的梯度
        
        # 反向传播通过时间
        dhnext = np.zeros((batch_size, self.hidden_size))
        
        for t in reversed(range(seq_len)):
            # 输出层梯度
            dWhy += np.dot(self.h[t+1].T, dy)
            dby += np.sum(dy, axis=0, keepdims=True)
            
            # 隐藏层梯度
            dh = np.dot(dy, self.Why.T) + dhnext
            
            # tanh梯度
            dh_raw = self.tanh_derivative(self.h[t+1]) * dh
            
            # 偏置梯度
            dbh += np.sum(dh_raw, axis=0, keepdims=True)
            
            # 权重梯度
            dWxh += np.dot(self.x[t].T, dh_raw)
            dWhh += np.dot(self.h[t].T, dh_raw)
            
            # 传递给下一个时间步
            dhnext = np.dot(dh_raw, self.Whh.T)
            
            # 如果不是最后一个时间步，计算y的梯度
            if t > 0:
                dy = np.zeros_like(self.y[t])
        
        # 梯度裁剪（防止梯度爆炸）
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -5, 5, out=dparam)
        
        # 更新权重
        self.Wxh -= self.learning_rate * dWxh
        self.Whh -= self.learning_rate * dWhh
        self.Why -= self.learning_rate * dWhy
        self.bh -= self.learning_rate * dbh
        self.by -= self.learning_rate * dby
    
    def train(self, X, y, epochs=100):
        """训练模型（带反向传播）"""
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # 计算损失
            loss = -np.mean(np.sum(y * np.log(y_pred[-1] + 1e-8), axis=1))
            
            # 反向传播
            self.backward(y)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """预测"""
        y_pred = self.forward(X)
        return np.argmax(y_pred[-1], axis=1)

def create_dataset():
    """创建简单数据集"""
    data = {
        'text': [
            '这个产品很好',
            '质量太差了',
            '服务不错',
            '价格太贵',
            '物流很快',
            '包装精美',
            '客服耐心',
            '商品有瑕疵',
            '性价比高',
            '发货太慢',
            '质量很好',
            '服务态度差',
            '价格合理',
            '物流太慢',
            '包装破损',
            '客服态度好',
            '产品很棒',
            '质量差',
            '服务很好',
            '价格便宜',
            '物流快',
            '包装好',
            '客服好',
            '商品好',
            '性价比低',
            '发货快',
            '质量差',
            '服务差',
            '价格高',
            '物流慢',
            '包装差',
            '客服差'
        ],
        'label': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 
                 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0]  # 1: 正面, 0: 负面
    }
    return pd.DataFrame(data)

def preprocess_text(text):
    """文本预处理"""
    words = jieba.lcut(text)
    return " ".join(words)

def text_to_sequence(text, vocab, max_len=5):
    """将文本转换为序列"""
    words = text.split()
    sequence = []
    for word in words[:max_len]:
        if word in vocab:
            sequence.append(vocab[word])
        else:
            sequence.append(0)
    
    # 填充到固定长度
    while len(sequence) < max_len:
        sequence.append(0)
    
    return sequence

def main():
    """主函数"""
    # 1. 数据准备
    print("=== 数据准备 ===")
    df = create_dataset()
    print(f"数据集大小: {len(df)}")
    
    # 2. 文本预处理
    print("\n=== 文本预处理 ===")
    df['processed'] = df['text'].apply(preprocess_text)
    print("预处理示例:")
    for i in range(3):
        print(f"原文: {df['text'].iloc[i]}")
        print(f"处理后: {df['processed'].iloc[i]}")
    
    # 3. 构建词汇表
    print("\n=== 构建词汇表 ===")
    all_words = []
    for text in df['processed']:
        all_words.extend(text.split())
    
    vocab = {'<PAD>': 0}
    for word in set(all_words):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    print("词汇表示例:", dict(list(vocab.items())[:10]))
    
    # 4. 转换为序列
    print("\n=== 序列转换 ===")
    max_len = 5
    sequences = []
    for text in df['processed']:
        seq = text_to_sequence(text, vocab, max_len)
        sequences.append(seq)
    
    # 转换为one-hot编码
    X = np.zeros((len(sequences), max_len, vocab_size))
    for i, seq in enumerate(sequences):
        for j, word_id in enumerate(seq):
            X[i, j, word_id] = 1
    
    y = df['label'].values
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1
    
    print("标签转换示例:")
    for i in range(5):
        print(f"原始标签: {y[i]} -> One-hot: {y_onehot[i]}")
    
    # 5. 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.25, random_state=42
    )
    
    # 调整维度为 (seq_len, batch_size, vocab_size)
    X_train = X_train.transpose(1, 0, 2)
    X_test = X_test.transpose(1, 0, 2)
    
    print(f"\n训练集形状: {X_train.shape}")
    print(f"测试集形状: {X_test.shape}")
    
    # 6. 训练RNN模型
    print("\n=== RNN模型训练（带反向传播）===")
    hidden_size = 16  # 增加隐藏层大小
    output_size = 2
    
    model = SimpleRNN(vocab_size, hidden_size, output_size, learning_rate=0.1)
    model.train(X_train, y_train, epochs=100)
    
    # 7. 模型评估
    print("\n=== 模型评估 ===")
    y_pred = model.predict(X_test)
    y_test_labels = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_test_labels, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 8. 预测新文本
    print("\n=== 新文本预测 ===")
    new_texts = [
        '这个商品质量很好',
        '服务态度太差了',
        '价格合理质量不错',
        '物流太慢了',
        '客服很耐心'
    ]
    
    for text in new_texts:
        processed = preprocess_text(text)
        seq = text_to_sequence(processed, vocab, max_len)
        
        # 转换为one-hot
        x = np.zeros((max_len, 1, vocab_size))
        for j, word_id in enumerate(seq):
            x[j, 0, word_id] = 1
        
        # 预测
        result = model.predict(x)[0]
        sentiment = "正面" if result == 1 else "负面"
        
        print(f"文本: {text}")
        print(f"预处理: {processed}")
        print(f"预测: {sentiment}")
        print()

if __name__ == "__main__":
    main() 