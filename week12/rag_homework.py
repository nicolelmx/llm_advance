"""
RAG文档翻译作业
"""

import math
import re
import json
from typing import List, Dict, Tuple, Optional
from collections import Counter
import random

class DocumentProcessor:
    """文档处理器 - 负责文档的预处理和向量化"""
    
    def __init__(self):
        self.vocabulary = {}
        self.stop_words = {'的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def preprocess_text(self, text: str) -> List[str]:
        """文本预处理"""
        # 转换为小写，去除标点符号
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # 分词
        words = text.split()
        # 移除停用词
        words = [word for word in words if word not in self.stop_words and len(word) > 1]
        return words
    
    def build_vocabulary(self, documents: List[str]):
        """构建词汇表"""
        word_counts = Counter()
        
        for doc in documents:
            words = self.preprocess_text(doc)
            word_counts.update(words)
        
        # 构建词汇表，过滤低频词
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.most_common()) if count >= 1}
        print(f"构建词汇表，包含 {len(self.vocabulary)} 个词")
    
    def tf_idf_encoding(self, documents: List[str]) -> List[List[float]]:
        """TF-IDF编码"""
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

class VectorDatabase:
    """向量数据库 - 存储和检索文档向量"""
    
    def __init__(self):
        self.documents = []
        self.vectors = []
        self.metadata = []
    
    def add_documents(self, documents: List[str], vectors: List[List[float]], metadata: List[Dict] = None):
        """添加文档到向量数据库"""
        self.documents.extend(documents)
        self.vectors.extend(vectors)
        
        if metadata:
            self.metadata.extend(metadata)
        else:
            # 默认元数据
            for i, doc in enumerate(documents):
                self.metadata.append({
                    'id': len(self.documents) - len(documents) + i,
                    'language': self.detect_language(doc),
                    'length': len(doc),
                    'word_count': len(doc.split())
                })
    
    def detect_language(self, text: str) -> str:
        """简单的语言检测"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return 'chinese'
        elif english_chars > chinese_chars:
            return 'english'
        else:
            return 'mixed'
    
    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """检索最相似的文档"""
        similarities = []
        
        for i, doc_vector in enumerate(self.vectors):
            # 计算余弦相似度
            similarity = self.cosine_similarity(query_vector, doc_vector)
            similarities.append((self.documents[i], similarity, self.metadata[i]))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
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

class TranslationMemory:
    """翻译记忆库 - 存储翻译对"""
    
    def __init__(self):
        self.translations = {}
        self.reverse_translations = {}
    
    def add_translation(self, source: str, target: str, source_lang: str, target_lang: str):
        """添加翻译对"""
        key = f"{source_lang}_{target_lang}"
        if key not in self.translations:
            self.translations[key] = {}
            self.reverse_translations[key] = {}
        
        self.translations[key][source] = target
        self.reverse_translations[key][target] = source
    
    def get_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """获取翻译"""
        key = f"{source_lang}_{target_lang}"
        if key in self.translations:
            return self.translations[key].get(text)
        return None
    
    def find_similar_translation(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """查找相似翻译（基于词汇重叠）"""
        key = f"{source_lang}_{target_lang}"
        if key not in self.translations:
            return None
        
        text_words = set(text.lower().split())
        best_match = None
        best_score = 0
        
        for source_text, target_text in self.translations[key].items():
            source_words = set(source_text.lower().split())
            overlap = len(text_words.intersection(source_words))
            if overlap > best_score:
                best_score = overlap
                best_match = target_text
        
        return best_match if best_score > 0 else None

class RAGTranslator:
    """RAG翻译器 - 基于检索增强生成的翻译系统"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_db = VectorDatabase()
        self.translation_memory = TranslationMemory()
        self.is_trained = False
    
    def train(self, parallel_corpus: List[Tuple[str, str]], source_lang: str = 'chinese', target_lang: str = 'english'):
        """训练翻译系统"""
        print("开始训练RAG翻译系统...")
        
        # 准备训练数据
        source_docs = [pair[0] for pair in parallel_corpus]
        target_docs = [pair[1] for pair in parallel_corpus]
        
        # 构建词汇表
        all_docs = source_docs + target_docs
        self.processor.build_vocabulary(all_docs)
        
        # 计算文档向量
        source_vectors = self.processor.tf_idf_encoding(source_docs)
        target_vectors = self.processor.tf_idf_encoding(target_docs)
        
        # 添加到向量数据库
        self.vector_db.add_documents(source_docs, source_vectors)
        self.vector_db.add_documents(target_docs, target_vectors)
        
        # 构建翻译记忆库
        for source, target in parallel_corpus:
            self.translation_memory.add_translation(source, target, source_lang, target_lang)
            self.translation_memory.add_translation(target, source, target_lang, source_lang)
        
        self.is_trained = True
        print(f"训练完成！共处理 {len(parallel_corpus)} 个翻译对")
    
    def translate(self, text: str, source_lang: str = 'chinese', target_lang: str = 'english') -> str:
        """翻译文本"""
        if not self.is_trained:
            return "系统未训练，请先调用train()方法"
        
        # 1. 检查翻译记忆库
        direct_translation = self.translation_memory.get_translation(text, source_lang, target_lang)
        if direct_translation:
            return f"[直接翻译] {direct_translation}"
        
        # 2. 查找相似翻译
        similar_translation = self.translation_memory.find_similar_translation(text, source_lang, target_lang)
        if similar_translation:
            return f"[相似翻译] {similar_translation}"
        
        # 3. 使用RAG进行翻译
        # 计算查询向量
        query_vector = self.processor.tf_idf_encoding([text])[0]
        
        # 检索相关文档
        if source_lang == 'chinese':
            # 检索中文文档
            relevant_docs = self.vector_db.search(query_vector, top_k=3)
            # 找到对应的英文翻译
            translations = []
            for doc, similarity, metadata in relevant_docs:
                if metadata['language'] == 'chinese':
                    # 查找对应的英文翻译
                    for source, target in self.translation_memory.translations.get(f"{source_lang}_{target_lang}", {}).items():
                        if source == doc:
                            translations.append((target, similarity))
                            break
            
            if translations:
                # 选择相似度最高的翻译
                best_translation = max(translations, key=lambda x: x[1])
                return f"[RAG翻译] {best_translation[0]}"
        
        return "[无匹配翻译] 未找到合适的翻译"
    
    def batch_translate(self, texts: List[str], source_lang: str = 'chinese', target_lang: str = 'english') -> List[str]:
        """批量翻译"""
        results = []
        for text in texts:
            translation = self.translate(text, source_lang, target_lang)
            results.append(translation)
        return results

def create_sample_corpus() -> List[Tuple[str, str]]:
    """创建样本平行语料库"""
    corpus = [
        ("人工智能是计算机科学的一个分支", "Artificial intelligence is a branch of computer science"),
        ("机器学习使计算机能够自动学习", "Machine learning enables computers to learn automatically"),
        ("深度学习基于神经网络进行学习", "Deep learning is based on neural networks for learning"),
        ("自然语言处理研究计算机理解人类语言", "Natural language processing studies how computers understand human language"),
        ("计算机视觉让机器能够理解图像", "Computer vision enables machines to understand images"),
        ("数据挖掘从大量数据中发现模式", "Data mining discovers patterns from large amounts of data"),
        ("推荐系统帮助用户找到感兴趣的内容", "Recommendation systems help users find interesting content"),
        ("算法是解决问题的步骤序列", "Algorithms are sequences of steps to solve problems"),
        ("编程语言是人与计算机交流的工具", "Programming languages are tools for human-computer communication"),
        ("软件开发需要系统性的方法", "Software development requires systematic approaches")
    ]
    return corpus

def create_test_documents() -> List[str]:
    """创建测试文档"""
    documents = [
        "人工智能技术在各个领域都有广泛应用",
        "机器学习算法需要大量的训练数据",
        "深度学习在图像识别方面表现出色",
        "自然语言处理包括文本分析和语言理解",
        "计算机视觉在自动驾驶中发挥重要作用",
        "数据科学结合了统计学和计算机科学",
        "云计算提供了灵活的计算资源",
        "物联网连接了各种智能设备",
        "区块链技术保证了数据的安全性",
        "量子计算有望解决复杂计算问题"
    ]
    return documents

def main():
    """主函数 - 演示RAG翻译系统"""
    print("RAG文档翻译作业演示")
    print("=" * 60)
    
    # 1. 创建翻译系统
    translator = RAGTranslator()
    
    # 2. 准备训练数据
    print("\n1. 准备训练数据")
    print("-" * 40)
    parallel_corpus = create_sample_corpus()
    print(f"准备了 {len(parallel_corpus)} 个中英翻译对")
    
    # 3. 训练系统
    print("\n2. 训练RAG翻译系统")
    print("-" * 40)
    translator.train(parallel_corpus, 'chinese', 'english')
    
    # 4. 测试翻译
    print("\n3. 测试翻译功能")
    print("-" * 40)
    
    test_texts = [
        "人工智能是计算机科学的一个分支",  # 直接匹配
        "机器学习算法很复杂",  # 相似匹配
        "区块链技术很有前景",  # RAG翻译
        "量子计算是未来技术"  # 无匹配
    ]
    
    for text in test_texts:
        translation = translator.translate(text, 'chinese', 'english')
        print(f"原文: {text}")
        print(f"译文: {translation}")
        print()
    
    # 5. 批量翻译测试
    print("\n4. 批量翻译测试")
    print("-" * 40)
    
    batch_texts = [
        "深度学习模型",
        "自然语言处理技术",
        "计算机视觉应用"
    ]
    
    batch_translations = translator.batch_translate(batch_texts, 'chinese', 'english')
    for original, translation in zip(batch_texts, batch_translations):
        print(f"{original} -> {translation}")
    
    # 6. 反向翻译测试
    print("\n5. 反向翻译测试")
    print("-" * 40)
    
    english_texts = [
        "Machine learning is powerful",
        "Deep learning models",
        "Computer vision systems"
    ]
    
    for text in english_texts:
        translation = translator.translate(text, 'english', 'chinese')
        print(f"原文: {text}")
        print(f"译文: {translation}")
        print()
    
    print("\n" + "=" * 60)
    print("RAG翻译系统演示完成！")

def explain_rag_concepts():
    """解释RAG核心概念"""
    print("RAG (Retrieval-Augmented Generation) 核心概念解释")
    print("=" * 60)
    
    print("\n1. RAG的基本原理:")
    print("  - Retrieval (检索): 从知识库中检索相关信息")
    print("  - Augmented (增强): 将检索到的信息与输入结合")
    print("  - Generation (生成): 基于增强的信息生成回答")
    
    print("\n2. RAG在翻译中的应用:")
    print("  - 检索阶段: 从翻译记忆库中查找相似或相关的翻译")
    print("  - 增强阶段: 将检索到的翻译信息与待翻译文本结合")
    print("  - 生成阶段: 基于检索信息生成最佳翻译")
    
    print("\n3. 本作业实现的关键组件:")
    print("  - DocumentProcessor: 文档预处理和向量化")
    print("  - VectorDatabase: 向量存储和相似度检索")
    print("  - TranslationMemory: 翻译记忆库管理")
    print("  - RAGTranslator: 整合所有组件的翻译系统")
    
    print("\n4. RAG的优势:")
    print("  - 能够利用历史翻译经验")
    print("  - 提供更准确和一致的翻译")
    print("  - 可以处理专业术语和特定领域文本")
    print("  - 支持增量学习和知识更新")
    
    print("\n5. 实际应用场景:")
    print("  - 专业文档翻译")
    print("  - 多语言内容管理")
    print("  - 翻译质量保证")
    print("  - 术语一致性维护")

if __name__ == "__main__":
    explain_rag_concepts()
    main()