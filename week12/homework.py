"""
📚第12周作业：写一个基于Rag的文档翻译系统，可自由发挥

# 作业参考资料

1. [RAG: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
2. [TF-IDF算法详解](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
3. [余弦相似度](https://en.wikipedia.org/wiki/Cosine_similarity)
4. [机器翻译技术综述](https://www.aclweb.org/anthology/2020.acl-main.1/)
"""
import math
import re
from typing import List, Tuple, Counter


class RAGTranslator:
    """RAG翻译器 - 基于检索增强生成的翻译系统"""

    def __init__(self):
        self.vocabulary = {}
        self.stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也',
                           '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 'the', 'a',
                           'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        self.documents = []
        self.vectors = []
        self.metadata = []
        self.translations = {}
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 转换为小写，去除标点符号
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # 分词
        words = text.split()
        # 移除停用词
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return words

    def get_documents_vectors(self, documents: List[str]) -> list:
        # 计算文档向量
        vectors = []

        # 计算每个文档的词频
        doc_word_counts = []
        for doc in documents:
            words = self.preprocess_text(doc)
            word_count = Counter(words)
            doc_word_counts.append(word_count)

        # 计算TF-IDF
        for word_count in doc_word_counts:
            vector = []
            for word in self.vocabulary:
                # TF: 词频
                tf = word_count.get(word, 0) / len(word_count) if word_count else 0

                # IDF: 逆文档频率
                doc_count = sum(1 for wc in doc_word_counts if word in wc)
                idf = math.log(len(doc_word_counts) / (doc_count + 1))

                # TF-IDF
                tf_idf = tf * idf
                vector.append(tf_idf)

            vectors.append(vector)

        return vectors

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def train(self, parallel_corpus: List[Tuple[str, str]]):
        source_docs = [pair[0] for pair in parallel_corpus]
        target_docs = [pair[1] for pair in parallel_corpus]
        self.documents = source_docs + target_docs

        word_count = Counter()
        for doc in self.documents:
            words = self.preprocess_text(doc)
            word_count.update(words)

        # 构建词汇表，过滤低频词
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_count.most_common()) if count >= 1}

        # 计算文档向量
        vectors = self.get_documents_vectors(self.documents)
        self.vectors.extend(vectors)
        # 默认元数据
        for i, doc in enumerate(self.documents):
            self.metadata.append({
                'id': i,
                'language': 'chinese' if doc in source_docs else 'english',
                'length': len(doc),
                'word_count': len(doc.split())
            })
        # 构建翻译记忆库 添加翻译对
        self.translations = {'chinese_english': {}, 'english_chinese': {}}
        for source_doc, target_doc in parallel_corpus:
            self.translations['chinese_english'][source_doc] = target_doc
            self.translations['english_chinese'][target_doc] = source_doc

    def translate(self, text: str, source_lang: str = 'chinese', target_lang: str = 'english') -> str:
        # 1. 检查翻译记忆库
        lang_key = f"{source_lang}_{target_lang}"
        if text in self.translations[lang_key]:
            return self.translations[lang_key][text]

        # 2. 查找相似翻译
        text_words = set(text.lower().split())
        best_score = 0
        best_match = None
        for source_doc, target_doc in self.translations[lang_key].items():
            source_words = set(source_doc.lower().split())
            score = len(text_words.intersection(source_words))
            if score > best_score:
                best_score = score
                best_match = target_doc
        if best_match:
            return best_match

        # 3. 使用RAG进行翻译
        # 计算查询向量
        query_vector = self.get_documents_vectors([text])[0]

        # 检索相关文档
        if source_lang == 'chinese':
            # 检索中文文档
            similarities = []
            for i, doc_vector in enumerate(self.vectors):
                # 计算余弦相似度
                similarity = self.cosine_similarity(query_vector, doc_vector)
                if self.metadata[i]['language'] == source_lang:
                    similarities.append((self.documents[i], similarity, self.metadata[i]))
            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            relevant_docs = similarities[:3]

            # 找到对应的英文翻译
            for doc, similarity, metadata in relevant_docs:
                # 查找对应的英文翻译
                if doc in self.translations.get(lang_key):
                    target = self.translations[lang_key][doc]
                    return f"[RAG翻译] {target}"

        return "[无匹配翻译] 未找到合适的翻译"


corpus = [
    ("人工智能是计算机科学的一个分支", "Artificial intelligence is a branch of computer science"),
    ("机器学习使计算机能够自动学习", "Machine learning enables computers to learn automatically"),
    ("深度学习基于神经网络进行学习", "Deep learning is based on neural networks for learning"),
    ("自然语言处理研究计算机理解人类语言",
     "Natural language processing studies how computers understand human language"),
    ("计算机视觉让机器能够理解图像", "Computer vision enables machines to understand images"),
    ("数据挖掘从大量数据中发现模式", "Data mining discovers patterns from large amounts of data"),
    ("推荐系统帮助用户找到感兴趣的内容", "Recommendation systems help users find interesting content"),
    ("算法是解决问题的步骤序列", "Algorithms are sequences of steps to solve problems"),
    ("编程语言是人与计算机交流的工具", "Programming languages are tools for human-computer communication"),
    ("软件开发需要系统性的方法", "Software development requires systematic approaches")
]
test_texts = [
    "人工智能是计算机科学的一个分支",  # 直接匹配
    "机器学习算法很复杂",  # 相似匹配
    "区块链技术很有前景",  # RAG翻译
    "量子计算是未来技术"  # 无匹配
]


if __name__ == '__main__':
    translator = RAGTranslator()
    translator.train(corpus)
    for text in test_texts:
        translation = translator.translate(text, 'chinese', 'english')
        print(f"原文: {text}")
        print(f"译文: {translation}")
