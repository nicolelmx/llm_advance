
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
    def __init__(self, vocab_size, hidden_size, output_size, learning_rate=0.01):
        """简单RNN网络"""
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # 初始化权重
        self.Wxh = np.random.randn(vocab_size, hidden_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(hidden_size, output_size) * 0.01
        self.bh = np.zeros((1, hidden_size))
        self.by = np.zeros((1, output_size))
    
    def tanh(self, x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    def softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        """前向传播"""
        seq_len, batch_size, _ = X.shape
        self.h = np.zeros((seq_len + 1, batch_size, self.hidden_size))
        self.y = np.zeros((seq_len, batch_size, self.output_size))
        
        for t in range(seq_len):
            # TODO 1: 完成RNN前向传播计算
            # 隐藏层: h_t = tanh(Wxh * x_t + Whh * h_{t-1} + bh)
            # 提示：使用np.dot()进行矩阵乘法，self.h[t]是上一时刻的隐藏状态
            self.h[t+1] = None  # 请完成这行代码
            
            # 输出层: y_t = softmax(Why * h_t + by)
            # 提示：使用当前时刻的隐藏状态self.h[t+1]计算输出
            self.y[t] = None  # 请完成这行代码
        
        return self.y
    
    def train(self, X, y, epochs=50):
        """训练模型（简化版 - 仅前向传播，无权重更新）"""
        print("注意：这个简化版本只计算损失，不更新权重，所以loss不会下降")
        print("这是为了帮助理解RNN的前向传播过程")
        print("要看到真正的训练效果，请参考改进版")
        print()
        
        for epoch in range(epochs):
            # 前向传播
            y_pred = self.forward(X)
            
            # TODO 2: 计算交叉熵损失
            # 提示：使用 y * np.log(y_pred[-1] + 1e-8) 计算交叉熵
            # 然后使用 np.sum() 和 np.mean() 计算平均损失
            loss = None  # 请完成这行代码
            
            if epoch % 10 == 0:
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
            '客服态度好'
        ],
        'label': [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]  # 1: 正面, 0: 负面
    }
    return pd.DataFrame(data)

def preprocess_text(text):
    """文本预处理"""
    # TODO 3: 完成文本预处理
    # 提示：使用jieba.lcut()进行中文分词，然后用空格连接
    # 返回处理后的文本字符串
    # return None  # 请完成这行代码
    text = text.lower()  # 转小写
    words = jieba.lcut(text)  # 中文分词
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
    
    # 3. 构建词汇表
    print("=== 构建词汇表 ===")
    all_words = []
    for text in df['processed']:
        all_words.extend(text.split())
    
    vocab = {'<PAD>': 0}
    for word in set(all_words):
        if word not in vocab:
            vocab[word] = len(vocab)
    
    vocab_size = len(vocab)
    print(f"词汇表大小: {vocab_size}")
    
    # 4. 转换为序列
    print("=== 序列转换 ===")
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
    
    # 5. 数据集分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_onehot, test_size=0.25, random_state=42
    )
    
    # 调整维度为 (seq_len, batch_size, vocab_size)
    X_train = X_train.transpose(1, 0, 2)
    X_test = X_test.transpose(1, 0, 2)
    
    # 6. 训练RNN模型
    print("\n=== RNN模型训练 ===")
    hidden_size = 8
    output_size = 2
    
    model = SimpleRNN(vocab_size, hidden_size, output_size, learning_rate=0.01)
    model.train(X_train, y_train, epochs=50)
    
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
        '价格合理质量不错'
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
        print(f"预测: {sentiment}")
        print()

if __name__ == "__main__":
    main() 