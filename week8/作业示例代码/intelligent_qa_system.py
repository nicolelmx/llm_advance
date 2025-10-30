#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能问答系统 - NLP核心任务进阶课程作业参考代码
结合序列标注（NER）和文本匹配技术实现智能客服系统

作者: NLP课程组
版本: 1.0
"""

import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers未安装，将使用简单的向量匹配")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from torchcrf import CRF
except ImportError:
    try:
        from TorchCRF import CRF
    except ImportError:
        logger.warning("CRF库未安装，请安装torchcrf或TorchCRF")
        CRF = None

# ==================== 数据结构定义 ====================

@dataclass
class Entity:
    """实体类"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 0.0

@dataclass
class FAQItem:
    """FAQ项目"""
    id: int
    question: str
    answer: str
    category: str = ""

@dataclass
class QueryResult:
    """查询结果"""
    answer: str
    entities: List[Entity]
    similar_questions: List[Tuple[str, float]]
    confidence: float

# ==================== 数据准备模块 ====================

class DataGenerator:
    """数据生成器 - 生成训练数据和FAQ数据"""
    
    def __init__(self):
        # 实体词典
        self.products = ["iPhone 15", "小米手机", "华为P60", "iPad", "MacBook", "小米平板", "华为笔记本"]
        self.orders = [f"ORD{str(i).zfill(6)}" for i in range(100000, 100100)]
        self.times = ["昨天", "今天", "明天", "上周", "本月", "2024年1月", "春节前", "双十一"]
        self.locations = ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "武汉"]
        self.prices = ["1000元", "￥2000", "3000块", "五千元", "1万元", "￥999", "2999元"]
        
        # 问题模板
        self.question_templates = [
            "我的{product}订单{order}什么时候能到{location}？",
            "{product}现在{price}能买到吗？",
            "订单{order}{time}能发货吗？",
            "我{time}在{location}买的{product}怎么还没到？",
            "{product}的价格是{price}吗？",
            "能帮我查一下订单{order}的物流信息吗？",
            "{location}有{product}的实体店吗？",
            "{time}下单的{product}什么时候能到？"
        ]
        
        # FAQ数据
        self.faq_data = [
            {"question": "订单什么时候能到？", "answer": "订单一般3-5个工作日送达，具体时间请查看物流信息。", "category": "物流"},
            {"question": "如何查询订单状态？", "answer": "您可以在订单页面输入订单号查询，或联系客服帮您查询。", "category": "订单"},
            {"question": "支持退换货吗？", "answer": "支持7天无理由退换货，商品需保持原包装完好。", "category": "售后"},
            {"question": "有什么优惠活动？", "answer": "目前有新用户注册送券活动，关注我们获取最新优惠信息。", "category": "优惠"},
            {"question": "配送费怎么算？", "answer": "订单满99元免配送费，不满99元收取8元配送费。", "category": "配送"},
            {"question": "支持货到付款吗？", "answer": "支持货到付款，但需要额外收取3元手续费。", "category": "支付"},
            {"question": "如何联系客服？", "answer": "您可以通过在线客服、电话400-123-4567或邮箱联系我们。", "category": "客服"},
            {"question": "商品有质量问题怎么办？", "answer": "如有质量问题，请及时联系客服，我们将为您安排退换货。", "category": "售后"},
            {"question": "可以指定送货时间吗？", "answer": "支持预约配送时间，请在下单时选择您方便的时间段。", "category": "配送"},
            {"question": "忘记密码怎么办？", "answer": "请点击登录页面的'忘记密码'，通过手机或邮箱重置密码。", "category": "账户"}
        ]
    
    def generate_training_data(self, num_samples: int = 50) -> List[Dict]:
        """生成训练数据"""
        data = []
        
        for i in range(num_samples):
            template = random.choice(self.question_templates)
            entities = []
            
            # 随机选择实体
            product = random.choice(self.products) if "{product}" in template else None
            order = random.choice(self.orders) if "{order}" in template else None
            time = random.choice(self.times) if "{time}" in template else None
            location = random.choice(self.locations) if "{location}" in template else None
            price = random.choice(self.prices) if "{price}" in template else None
            
            # 构建问题文本
            question = template
            if product:
                question = question.replace("{product}", product)
            if order:
                question = question.replace("{order}", order)
            if time:
                question = question.replace("{time}", time)
            if location:
                question = question.replace("{location}", location)
            if price:
                question = question.replace("{price}", price)
            
            # 标注实体位置
            current_pos = 0
            if product:
                start = question.find(product)
                if start != -1:
                    entities.append({
                        "text": product,
                        "label": "PRODUCT",
                        "start": start,
                        "end": start + len(product)
                    })
            
            if order:
                start = question.find(order)
                if start != -1:
                    entities.append({
                        "text": order,
                        "label": "ORDER",
                        "start": start,
                        "end": start + len(order)
                    })
            
            if time:
                start = question.find(time)
                if start != -1:
                    entities.append({
                        "text": time,
                        "label": "TIME",
                        "start": start,
                        "end": start + len(time)
                    })
            
            if location:
                start = question.find(location)
                if start != -1:
                    entities.append({
                        "text": location,
                        "label": "LOCATION",
                        "start": start,
                        "end": start + len(location)
                    })
            
            if price:
                start = question.find(price)
                if start != -1:
                    entities.append({
                        "text": price,
                        "label": "PRICE",
                        "start": start,
                        "end": start + len(price)
                    })
            
            data.append({
                "id": i + 1,
                "text": question,
                "entities": entities
            })
        
        return data
    
    def get_faq_data(self) -> List[Dict]:
        """获取FAQ数据"""
        return [{"id": i+1, **faq} for i, faq in enumerate(self.faq_data)]
    
    def save_data(self, train_data: List[Dict], faq_data: List[Dict], output_dir: str = "data"):
        """保存数据到文件"""
        Path(output_dir).mkdir(exist_ok=True)
        
        dataset = {
            "questions": train_data,
            "faq": faq_data
        }
        
        with open(f"{output_dir}/dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到 {output_dir}/dataset.json")

# ==================== NER模型模块 ====================

class NERModel:
    """命名实体识别模型"""
    
    def __init__(self, model_type: str = 'bilstm_crf'):
        self.model_type = model_type
        self.model = None
        self.word_to_ix = {}
        self.tag_to_ix = {}
        self.ix_to_tag = {}
        
        # 标签集合
        self.labels = ['O', 'B-PRODUCT', 'I-PRODUCT', 'B-ORDER', 'I-ORDER', 
                      'B-TIME', 'I-TIME', 'B-LOCATION', 'I-LOCATION', 
                      'B-PRICE', 'I-PRICE']
        
        for i, label in enumerate(self.labels):
            self.tag_to_ix[label] = i
            self.ix_to_tag[i] = label
    
    def _prepare_data(self, data: List[Dict]) -> List[Tuple[List[str], List[str]]]:
        """准备训练数据"""
        training_data = []
        
        for item in data:
            text = item['text']
            entities = item['entities']
            
            # 字符级别标注
            chars = list(text)
            labels = ['O'] * len(chars)
            
            # 标注实体
            for entity in entities:
                start, end = entity['start'], entity['end']
                label = entity['label']
                
                if start < len(labels):
                    labels[start] = f'B-{label}'
                    for i in range(start + 1, min(end, len(labels))):
                        labels[i] = f'I-{label}'
            
            training_data.append((chars, labels))
        
        return training_data
    
    def _build_vocab(self, training_data: List[Tuple[List[str], List[str]]]):
        """构建词汇表"""
        for chars, labels in training_data:
            for char in chars:
                if char not in self.word_to_ix:
                    self.word_to_ix[char] = len(self.word_to_ix)
        
        # 添加特殊符号
        self.word_to_ix['<UNK>'] = len(self.word_to_ix)
        self.word_to_ix['<PAD>'] = len(self.word_to_ix)
    
    def train(self, train_data: List[Dict], val_data: List[Dict] = None):
        """训练模型"""
        logger.info(f"开始训练{self.model_type}模型...")
        
        # 准备数据
        training_data = self._prepare_data(train_data)
        self._build_vocab(training_data)
        
        if self.model_type == 'bilstm_crf':
            self._train_bilstm_crf(training_data)
        else:
            logger.warning(f"不支持的模型类型: {self.model_type}")
    
    def _train_bilstm_crf(self, training_data: List[Tuple[List[str], List[str]]]):
        """训练BiLSTM+CRF模型"""
        if CRF is None:
            logger.error("CRF库未安装，无法训练BiLSTM+CRF模型")
            return
        
        # 简化版BiLSTM+CRF模型
        class SimpleBiLSTMCRF(nn.Module):
            def __init__(self, vocab_size, tag_to_ix, embedding_dim=50, hidden_dim=100):
                super().__init__()
                self.embedding_dim = embedding_dim
                self.hidden_dim = hidden_dim
                self.vocab_size = vocab_size
                self.tag_to_ix = tag_to_ix
                self.tagset_size = len(tag_to_ix)
                
                self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, batch_first=True)
                self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
                self.crf = CRF(self.tagset_size)
            
            def forward(self, sentence):
                embeds = self.word_embeds(sentence).unsqueeze(0)
                lstm_out, _ = self.lstm(embeds)
                emissions = self.hidden2tag(lstm_out)
                return self.crf.decode(emissions.transpose(0, 1))[0]
            
            def neg_log_likelihood(self, sentence, tags):
                embeds = self.word_embeds(sentence).unsqueeze(0)
                lstm_out, _ = self.lstm(embeds)
                emissions = self.hidden2tag(lstm_out)
                return -self.crf(emissions.transpose(0, 1), tags.unsqueeze(1))
        
        # 创建模型
        self.model = SimpleBiLSTMCRF(len(self.word_to_ix), self.tag_to_ix)
        optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        
        # 训练循环（简化版）
        for epoch in range(10):
            total_loss = 0
            for chars, labels in training_data[:10]:  # 限制训练样本数量
                # 准备输入
                char_ids = torch.tensor([self.word_to_ix.get(c, self.word_to_ix['<UNK>']) for c in chars])
                label_ids = torch.tensor([self.tag_to_ix[l] for l in labels])
                
                # 前向传播
                self.model.zero_grad()
                loss = self.model.neg_log_likelihood(char_ids, label_ids)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
        logger.info("BiLSTM+CRF模型训练完成")
    
    def predict(self, text: str) -> List[Entity]:
        """预测实体"""
        if self.model is None:
            logger.warning("模型未训练，返回空结果")
            return []
        
        chars = list(text)
        char_ids = torch.tensor([self.word_to_ix.get(c, self.word_to_ix['<UNK>']) for c in chars])
        
        with torch.no_grad():
            predicted_tags = self.model(char_ids)
        
        # 解析实体
        entities = []
        current_entity = None
        
        for i, tag_id in enumerate(predicted_tags):
            tag = self.ix_to_tag[tag_id]
            
            if tag.startswith('B-'):
                # 开始新实体
                if current_entity:
                    entities.append(current_entity)
                current_entity = Entity(
                    text=chars[i],
                    label=tag[2:],
                    start=i,
                    end=i+1,
                    confidence=0.8
                )
            elif tag.startswith('I-') and current_entity:
                # 继续当前实体
                current_entity.text += chars[i]
                current_entity.end = i + 1
            else:
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估模型性能"""
        # 简化版评估
        correct = 0
        total = 0
        
        for item in test_data[:5]:  # 限制评估样本
            predicted_entities = self.predict(item['text'])
            true_entities = item['entities']
            
            # 简单的精确匹配评估
            for pred_entity in predicted_entities:
                for true_entity in true_entities:
                    if (pred_entity.text == true_entity['text'] and 
                        pred_entity.label == true_entity['label']):
                        correct += 1
                        break
            
            total += len(true_entities)
        
        precision = correct / max(total, 1)
        return {
            'precision': precision,
            'recall': precision,  # 简化
            'f1': precision
        }

# ==================== 文本匹配模块 ====================

class TextMatcher:
    """文本相似度匹配器"""
    
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"已加载Sentence-BERT模型: {model_name}")
            except:
                logger.warning("无法加载Sentence-BERT模型，使用简单相似度计算")
                self.model = None
        else:
            self.model = None
    
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """编码文本为向量"""
        if self.model:
            return self.model.encode(texts)
        else:
            # 简单的词频向量
            from collections import Counter
            
            # 构建词汇表
            vocab = set()
            for text in texts:
                vocab.update(text)
            vocab = list(vocab)
            
            # 转换为向量
            vectors = []
            for text in texts:
                counter = Counter(text)
                vector = [counter.get(word, 0) for word in vocab]
                vectors.append(vector)
            
            return np.array(vectors)
    
    def find_similar(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        """找到最相似的文本"""
        if not candidates:
            return []
        
        # 编码所有文本
        all_texts = [query] + candidates
        embeddings = self.encode_texts(all_texts)
        
        # 计算相似度
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.model:
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        else:
            # 简单的余弦相似度
            def cosine_sim(a, b):
                dot_product = np.dot(a, b)
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                return dot_product / (norm_a * norm_b) if norm_a * norm_b != 0 else 0
            
            similarities = [cosine_sim(query_embedding[0], cand_emb) for cand_emb in candidate_embeddings]
        
        # 排序并返回top_k
        results = [(candidates[i], float(sim)) for i, sim in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def batch_search(self, queries: List[str], candidates: List[str]) -> List[List[Tuple[str, float]]]:
        """批量搜索"""
        return [self.find_similar(query, candidates) for query in queries]

# ==================== 智能问答系统 ====================

class IntelligentQASystem:
    """智能问答系统主类"""
    
    def __init__(self):
        self.ner_model = NERModel()
        self.text_matcher = TextMatcher()
        self.faq_database = []
        self.faq_questions = []
        
        logger.info("智能问答系统初始化完成")
    
    def load_faq(self, faq_data: List[Dict]):
        """加载FAQ数据库"""
        self.faq_database = [FAQItem(**item) for item in faq_data]
        self.faq_questions = [item.question for item in self.faq_database]
        logger.info(f"已加载{len(self.faq_database)}条FAQ数据")
    
    def train_models(self, train_data: List[Dict]):
        """训练模型"""
        logger.info("开始训练NER模型...")
        self.ner_model.train(train_data)
        logger.info("模型训练完成")
    
    def process_query(self, user_question: str) -> QueryResult:
        """处理用户查询"""
        logger.info(f"处理查询: {user_question}")
        
        # 1. 识别实体
        entities = self.ner_model.predict(user_question)
        logger.info(f"识别到{len(entities)}个实体: {[e.text for e in entities]}")
        
        # 2. 检索相关FAQ
        if self.faq_questions:
            similar_questions = self.text_matcher.find_similar(
                user_question, self.faq_questions, top_k=3
            )
        else:
            similar_questions = []
        
        # 3. 生成答案
        answer = self.generate_answer(entities, similar_questions)
        
        # 4. 计算置信度
        confidence = self._calculate_confidence(entities, similar_questions)
        
        return QueryResult(
            answer=answer,
            entities=entities,
            similar_questions=similar_questions,
            confidence=confidence
        )
    
    def generate_answer(self, entities: List[Entity], similar_questions: List[Tuple[str, float]]) -> str:
        """基于实体和相似问题生成答案"""
        if not similar_questions:
            return "抱歉，我没有找到相关的答案。请联系人工客服获得帮助。"
        
        # 找到最相似的问题
        best_question, similarity = similar_questions[0]
        
        # 找到对应的答案
        for faq_item in self.faq_database:
            if faq_item.question == best_question:
                answer = faq_item.answer
                
                # 根据实体个性化答案
                if entities:
                    entity_info = "、".join([f"{e.label}: {e.text}" for e in entities])
                    answer += f"\n\n检测到相关信息：{entity_info}"
                
                return answer
        
        return "抱歉，系统出现错误，请稍后再试。"
    
    def _calculate_confidence(self, entities: List[Entity], similar_questions: List[Tuple[str, float]]) -> float:
        """计算置信度"""
        if not similar_questions:
            return 0.0
        
        # 基于最高相似度和实体数量计算置信度
        max_similarity = similar_questions[0][1]
        entity_bonus = min(len(entities) * 0.1, 0.3)  # 实体越多置信度越高
        
        return min(max_similarity + entity_bonus, 1.0)

# ==================== 演示和测试 ====================

def main():
    """主函数 - 演示系统功能"""
    print("=" * 60)
    print("智能问答系统演示")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n1. 生成训练数据和FAQ数据...")
    data_generator = DataGenerator()
    train_data = data_generator.generate_training_data(30)  # 减少数据量以加快演示
    faq_data = data_generator.get_faq_data()
    
    # 保存数据
    data_generator.save_data(train_data, faq_data)
    print(f"生成了{len(train_data)}个训练样本和{len(faq_data)}个FAQ")
    
    # 2. 初始化系统
    print("\n2. 初始化智能问答系统...")
    qa_system = IntelligentQASystem()
    qa_system.load_faq(faq_data)
    
    # 3. 训练模型
    print("\n3. 训练NER模型...")
    qa_system.train_models(train_data)
    
    # 4. 测试查询
    print("\n4. 测试查询...")
    test_questions = [
        "我的iPhone 15订单ORD123456什么时候能到北京？",
        "小米手机现在2000元能买到吗？",
        "订单什么时候能到？",
        "支持退换货吗？",
        "华为P60在上海有实体店吗？"
    ]
    
    for question in test_questions:
        print(f"\n查询: {question}")
        result = qa_system.process_query(question)
        
        print(f"识别实体: {[(e.text, e.label) for e in result.entities]}")
        print(f"相似问题: {result.similar_questions[:2]}")
        print(f"置信度: {result.confidence:.2f}")
        print(f"答案: {result.answer}")
        print("-" * 50)
    
    # 5. 模型评估
    print("\n5. 模型评估...")
    test_data = train_data[:5]  # 使用部分训练数据作为测试
    metrics = qa_system.ner_model.evaluate(test_data)
    print(f"NER模型性能: {metrics}")
    
    print("\n系统演示完成！")

if __name__ == "__main__":
    main()
