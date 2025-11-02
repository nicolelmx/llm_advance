import os
import json
import numpy as np
from typing import List, Dict, Tuple
import jieba  # 中文词库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import time


class SimpleRAGDemo:
    """简单RAG系统演示类"""

    def __init__(self):
        self.documents = []  # 存储文档片段
        self.embeddings = []  # 存储向量表示
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2) # 1-gram 2-gram 单词 双词组合特征
        )
        self.is_fitted = False

    def preprocess_text(self, text: str) -> str:
        """文本预处理"""
        # 去除多余空白字符
        text = ' '.join(text.split())
        # 中文分词
        words = jieba.cut(text)
        return ' '.join(words)

    def chunk_document(self, text: str, chunk_size: int = 200, overlap: int = 50) -> List[str]:
        """
        文档分块策略演示
        
        Args:
            text: 原始文档文本
            chunk_size: 每个块的字符数
            overlap: 重叠字符数
            
        Returns:
            分块后的文档列表
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]

            # 避免在句子中间切断
            if end < len(text) and text[end] not in ['。', '！', '？', '\n']:
                # 寻找最近的句号
                last_period = chunk.rfind('。')
                if last_period > chunk_size // 2:  # 确保块不会太小
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - overlap

            if start >= len(text):
                break

        return [chunk for chunk in chunks if len(chunk.strip()) > 20]

    def add_documents(self, documents: List[str]):
        """添加文档到知识库"""
        print("📚 正在处理文档...")

        all_chunks = []
        for i, doc in enumerate(documents):
            print(f"处理文档 {i + 1}/{len(documents)}")

            # 预处理
            processed_doc = self.preprocess_text(doc) # 数据的清晰和分词

            # 分块
            chunks = self.chunk_document(processed_doc)
            all_chunks.extend(chunks)

            print(f"  - 生成了 {len(chunks)} 个文档块")

        self.documents = all_chunks
        print(f"✅ 总共处理了 {len(self.documents)} 个文档块")

        # 生成向量表示
        self._create_embeddings()

    def _create_embeddings(self):
        """创建文档的向量表示"""
        print("🔄 正在生成向量表示...")

        # 使用TF-IDF作为简单的embedding方法
        self.embeddings = self.vectorizer.fit_transform(self.documents)
        self.is_fitted = True

        print(f"✅ 生成了 {self.embeddings.shape[0]} 个向量，维度为 {self.embeddings.shape[1]}")

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回最相关的k个文档
            
        Returns:
            (文档, 相似度分数) 的列表
        """
        if not self.is_fitted:
            raise ValueError("请先添加文档到知识库")

        print(f"🔍 搜索查询: '{query}'")

        # 预处理查询 分词后的查询
        processed_query = self.preprocess_text(query)

        # 将查询转换为向量
        query_vector = self.vectorizer.transform([processed_query])

        # 计算相似度
        similarities = cosine_similarity(query_vector, self.embeddings)[0]

        # 获取最相似的文档
        top_indices = np.argsort(similarities)[::-1][:top_k] # 按照相似度索引进行排序

        results = []
        for i, idx in enumerate(top_indices):
            doc = self.documents[idx]
            score = similarities[idx]
            results.append((doc, score))
            print(f"  {i + 1}. 相似度: {score:.4f}")
            print(f"     内容: {doc[:100]}...")

        return results

    def generate_answer(self, query: str, context_docs: List[str]) -> str:
        """
        基于检索到的文档生成回答
        这里使用简单的模板方法，实际应用中会使用LLM
        """
        print("🤖 正在生成回答...")

        # 构建上下文
        context = "\n".join([f"参考资料{i + 1}: {doc}" for i, doc in enumerate(context_docs)])

        # 简单的回答生成逻辑（实际应用中应该使用LLM）
        answer = f"""基于提供的参考资料，针对问题"{query}"的回答：

参考上下文：
{context}

回答：根据以上参考资料，这个问题涉及到多个方面的内容。建议结合具体的上下文信息进行详细分析。

注意：这是一个简化的演示版本，实际的RAG系统会使用大语言模型来生成更准确和自然的回答。"""

        return answer

    def rag_pipeline(self, query: str, top_k: int = 3) -> str:
        """完整的RAG流程"""
        print("🚀 启动RAG流程")
        print("=" * 50)

        # 1. 检索相关文档
        search_results = self.search(query, top_k)

        # 2. 提取文档内容
        context_docs = [doc for doc, score in search_results if score > 0.1]

        if not context_docs:
            return "抱歉，没有找到相关的信息来回答您的问题。"

        # 3. 生成回答
        answer = self.generate_answer(query, context_docs)

        print("=" * 50)
        print("✅ RAG流程完成")

        return answer


def create_sample_documents() -> List[str]:
    """创建演示用的样本文档"""
    documents = [
        """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
        并生产出一种新的能以人类智能相似的方式做出反应的智能机器。人工智能的研究领域包括机器学习、
        自然语言处理、计算机视觉、专家系统等。近年来，深度学习技术的发展推动了人工智能的快速进步。
        """,

        """
        机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习。
        机器学习算法通过训练数据来构建数学模型，以便对新数据做出预测或决策。
        主要的机器学习方法包括监督学习、无监督学习和强化学习。常见的算法有线性回归、
        决策树、支持向量机、神经网络等。
        """,

        """
        自然语言处理（Natural Language Processing，NLP）是人工智能和语言学领域的分支学科。
        它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。
        NLP的主要任务包括词法分析、句法分析、语义分析、机器翻译、情感分析、
        问答系统等。近年来，基于Transformer的大语言模型在NLP领域取得了突破性进展。
        """,

        """
        深度学习是机器学习的一个子领域，它基于人工神经网络进行学习。
        深度学习模型通常包含多个隐藏层，能够学习数据的复杂表示。
        卷积神经网络（CNN）在计算机视觉领域表现出色，循环神经网络（RNN）
        和长短期记忆网络（LSTM）在序列数据处理方面很有效。
        Transformer架构的出现革命性地改变了自然语言处理领域。
        """,

        """
        RAG（Retrieval-Augmented Generation）是一种结合信息检索和文本生成的AI技术。
        它在生成回答前先从知识库中检索相关信息，然后基于检索到的信息生成更准确的回答。
        RAG技术可以有效解决大模型的知识截止时间限制和幻觉问题，
        在问答系统、智能客服、知识库查询等场景中有广泛应用。
        """
    ]

    return documents


def main():
    """主函数 - 演示RAG系统的完整流程"""
    print("🎯 基础RAG系统演示")
    print("=" * 60)

    # 1. 初始化RAG系统
    rag_system = SimpleRAGDemo()

    # 2. 准备样本文档
    print("📋 准备样本文档...")
    documents = create_sample_documents()

    # 3. 添加文档到知识库
    rag_system.add_documents(documents)

    print("\n" + "=" * 60)

    # 4. 演示查询
    queries = [
        "什么是人工智能？",
        "机器学习有哪些主要方法？",
        "RAG技术是如何工作的？",
        "深度学习和传统机器学习有什么区别？"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n📝 演示查询 {i}: {query}")
        print("-" * 40)

        # 执行RAG流程
        answer = rag_system.rag_pipeline(query)
        print(f"\n💡 生成的回答:\n{answer}")

        print("\n" + "=" * 60)

        # 添加延时，便于观察
        time.sleep(1)

    print("🎉 演示完成！")

    # 5. 交互式查询（可选）
    print("\n" + "=" * 60)
    print("💬 进入交互模式（输入 'quit' 退出）:")

    while True:
        user_query = input("\n请输入您的问题: ").strip()

        if user_query.lower() in ['quit', 'exit', '退出']:
            print("👋 再见！")
            break

        if user_query:
            try:
                answer = rag_system.rag_pipeline(user_query)
                print(f"\n💡 回答:\n{answer}")
            except Exception as e:
                print(f"❌ 处理查询时出错: {e}")


if __name__ == "__main__":
    # 安装必要的依赖
    try:
        import jieba
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError as e:
        print("❌ 缺少必要的依赖库，请安装:")
        print("pip install jieba scikit-learn numpy")
        exit(1)

    main()
