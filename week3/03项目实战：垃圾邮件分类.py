# 禁用警告信息
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 设置jieba不显示加载信息
jieba.setLogLevel(20)

# 简单的垃圾邮件数据集
data = {
    'text': [
        'Hi, you have won a lottery, click here to claim',  # 垃圾邮件
        'Lunch meeting tomorrow at 12pm?',  # 正常邮件
        'URGENT! Your account has been compromised!',  # 垃圾邮件
        'Can you please review the document?',  # 正常邮件
        'Free Viagra, cheap Cialis, order now!',  # 垃圾邮件
        '恭喜您中奖！请点击链接领取大奖！',  # 垃圾邮件
        '你好，下周的会议纪要发你了。',  # 正常邮件
        '【xx贷】急用钱？马上到账！',  # 垃圾邮件
        '周末一起去打球吗？',  # 正常邮件
        '发票，代开，增值税。'  # 垃圾邮件
    ],
    'label': [1, 0, 1, 0, 1, 1, 0, 1, 0, 1]  # 1: 垃圾邮件, 0: 正常邮件
}

df = pd.DataFrame(data)
print("原始数据:")
print(df)

# 简单的文本预处理
def preprocess_text(text):
    text = text.lower()  # 转小写
    words = jieba.lcut(text)  # 中文分词
    return " ".join(words)

# 预处理数据
df['processed'] = df['text'].apply(preprocess_text)
print("\n预处理后的数据:")
print(df[['text', 'label', 'processed']])

# 特征提取
vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(df['processed'])
y = df['label']

# 训练模型
model = MultinomialNB()
model.fit(X, y)

# 预测新邮件
new_emails = [
    'Can we reschedule our meeting?',
    '【投资理财】轻松月入过万！',
    'Your package has been delivered'
]

print("\n=== 预测结果 ===")
for email in new_emails:
    # 预处理
    processed = preprocess_text(email)
    # 特征提取
    features = vectorizer.transform([processed])
    # 预测
    prediction = model.predict(features)[0]
    result = "垃圾邮件" if prediction == 1 else "正常邮件"
    print(f"邮件: {email}")
    print(f"预测: {result}")
    print()