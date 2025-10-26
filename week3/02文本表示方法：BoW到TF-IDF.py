# 准备语料
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    '今天 天气 真好',
    '今天 天气 不好',
    '今天 心情 真好',
    '我 的 心情 不好'
]
# 假设已经分好词，用空格隔开

# 2. 词袋模型 (Bag-of-Words) 实现
# 1. 创建CountVectorizer实例
vectorizer_bow = CountVectorizer()

# 2. 学习词汇表并转换语料
X_bow = vectorizer_bow.fit_transform(corpus)

# 3. 查看结果
print("词袋模型 (BoW) 结果:")
print("词汇表:", vectorizer_bow.get_feature_names_out())
print("稀疏矩阵表示:\n", X_bow)
print("稠密矩阵表示:\n", X_bow.toarray())


# TF-IDF实现
# 1. 创建TfidfVectorizer实例
vectorizer_tfidf = TfidfVectorizer()

# 2. 学习词汇表并计算TF-IDF权重
X_tfidf = vectorizer_tfidf.fit_transform(corpus)

# 3. 查看结果
print("\nTF-IDF 结果:")
print("词汇表:", vectorizer_tfidf.get_feature_names_out())
print("TF-IDF权重矩阵 (稠密表示):\n", X_tfidf.toarray().round(2))

