import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

class BERTTeachingDemo:
    def __init__(self):
        """初始化BERT教学演示类"""
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"使用设备: {self.device}")

    def load_bert_model(self):
        """1. 加载预训练BERT模型"""
        print("=== 1. 加载BERT模型 ===")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

            # 加载BERT模型
            self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)

            # 加载MLM任务模型
            self.mlm_model = BertForMaskedLM.from_pretrained('bert-base-chinese').to(self.device)

            print("✓ BERT模型加载完成")
            print(f"模型参数量: {sum(p.numel() for p in self.bert_model.parameters()):,}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("💡 请检查网络连接或下载模型文件到本地")
            print("   离线模式: 设置环境变量 HF_HUB_OFFLINE=1")
            print("   或手动下载模型到: ~/.cache/huggingface/hub/")
            return False

    def demonstrate_tokenization(self):
        """2. 演示BERT分词"""
        print("\n=== 2. BERT分词演示 ===")

        text = "自然语言处理技术正在快速发展"

        # 基本分词
        tokens = self.tokenizer.tokenize(text)
        print(f"原始文本: {text}")
        print(f"分词结果: {tokens}")

        # 转换为ID add_special_tokens默认true，分词结果会包含特殊标记。[CLS]：句子的起始标记（位于句首）[SEP]：句子的分隔标记
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        print(f"输入ID: {input_ids}")
        print(f"特殊token: [CLS]={self.tokenizer.cls_token_id}, [SEP]={self.tokenizer.sep_token_id}")

    def demonstrate_attention_mask(self):
        """3. 演示注意力掩码"""
        print("\n=== 3. 注意力掩码演示 ===")

        texts = ["今天天气很好", "自然语言处理很有趣"]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

        print("批量编码结果:")
        print(f"输入ID形状: {encoded['input_ids'].shape}")
        print(f"注意力掩码: {encoded['attention_mask']}")

    def demonstrate_bert_forward(self):
        """4. 演示BERT前向传播"""
        print("\n=== 4. BERT前向传播演示 ===")

        text = "BERT模型能够学习丰富的语言表征"
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        print(f"输入文本: {text}")
        print(f"last_hidden_state形状: {outputs.last_hidden_state.shape}")
        print(f"pooler_output形状: {outputs.pooler_output.shape}")

        # [CLS] token的表征
        cls_embedding = outputs.last_hidden_state[0, 0, :]
        print(f"[CLS] token表征维度: {cls_embedding.shape}")

    def demonstrate_mlm_task(self):
        """5. 演示掩码语言模型任务"""
        print("\n=== 5. 掩码语言模型(MLM)演示 ===")

        text = "自然语言处理是人工智能的重要研究方向。"
        print(f"原始文本: {text}")

        # 使用tokenizer进行分词，然后创建掩码
        tokens = self.tokenizer.tokenize(text)
        print(f"分词结果: {tokens}")
        
        # 创建掩码输入
        masked_tokens = tokens.copy()
        mask_positions = []

        # 随机掩码15%的token（跳过特殊token）
        for i, token in enumerate(tokens):
            if token not in ['[CLS]', '[SEP]'] and np.random.random() < 0.15:
                masked_tokens[i] = "[MASK]"
                mask_positions.append(i)

        # 重新编码为完整句子
        masked_sentence = self.tokenizer.convert_tokens_to_string(masked_tokens)
        print(f"掩码后: {masked_sentence}")
        print(f"掩码位置: {mask_positions}")

        # 模型预测
        inputs = self.tokenizer(masked_sentence, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.mlm_model(**inputs)
            predictions = outputs.logits

        # 获取预测结果
        mask_token_index = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
        print('mask_token_index', mask_token_index)

        if len(mask_token_index) > 0:
            for i, pos in enumerate(mask_token_index):
                predicted_token_id = predictions[0, pos].argmax().item()
                predicted_token = self.tokenizer.decode(predicted_token_id)
                print(f"位置{pos.item()}: [MASK] -> {predicted_token}")
        else:
            print("本次运行没有生成掩码token，这是正常的随机行为")

    def create_classification_dataset(self):
        """创建文本分类数据集"""
        # 模拟新闻分类数据集
        texts = [
            "科技公司发布新款智能手机",  # 科技
            "政府出台新的经济政策",      # 政治
            "足球比赛精彩纷呈",          # 体育
            "电影票房创下新高",          # 娱乐
            "股市出现大幅波动",          # 财经
        ]
        labels = [0, 1, 2, 3, 4]  # 对应的标签

        return texts, labels

    def fine_tune_bert_classifier(self):
        """6. BERT微调实战：文本分类"""
        print("\n=== 6. BERT微调实战：新闻分类 ===")

        # 准备数据
        texts, labels = self.create_classification_dataset()
        labels = torch.tensor(labels).to(self.device)

        # 加载分类模型
        classifier = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese',
            num_labels=5
        ).to(self.device)

        # 编码数据
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                               return_tensors='pt').to(self.device)

        # 训练参数
        optimizer = AdamW(classifier.parameters(), lr=2e-5)
        num_epochs = 3
        batch_size = 2

        print("开始微调训练...")

        for epoch in range(num_epochs):
            classifier.train()

            # 前向传播
            outputs = classifier(**inputs, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            # probabilities = F.softmax(logits, dim=1)  # 转换为概率
            # predictions = torch.argmax(probabilities, dim=1)  # 预测标签索引
            # print(predictions)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            print(predictions)
            accuracy = (predictions == labels).float().mean()
            print(predictions)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.3f}, Accuracy: {accuracy.item():.3f}")

        print("✓ 微调完成")

    def compare_bert_vs_gpt(self):
        """7. BERT与GPT对比分析"""
        print("\n=== 7. BERT vs GPT 对比分析 ===")

        comparison = {
            "架构": {
                "BERT": "双向Transformer编码器",
                "GPT": "单向Transformer解码器"
            },
            "注意力机制": {
                "BERT": "全词关注（无掩码）",
                "GPT": "掩码自注意力"
            },
            "优势场景": {
                "BERT": "文本理解任务",
                "GPT": "文本生成任务"
            },
            "预训练任务": {
                "BERT": "MLM + NSP",
                "GPT": "自回归语言建模"
            },
            "参数量级": {
                "BERT": "Base版1.1亿参数",
                "GPT": "GPT-3达1750亿参数"
            }
        }

        print("BERT与GPT核心对比:")
        for aspect, models in comparison.items():
            print(f"{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")
            print()

    def run_all_demos(self):
        """运行所有BERT教学演示"""
        print("🚀 开始BERT教学演示\n")

        if not self.load_bert_model():
            print("\n⚠️  由于模型加载失败，跳过需要模型的演示")
            print("📖 您可以查看代码注释了解BERT的工作原理")
            return

        self.demonstrate_tokenization()
        self.demonstrate_attention_mask()
        self.demonstrate_bert_forward()
        self.demonstrate_mlm_task()
        self.fine_tune_bert_classifier()
        # self.compare_bert_vs_gpt()

        print("\n🎉 BERT教学演示完成！")

# 主程序
if __name__ == "__main__":
    demo = BERTTeachingDemo()
    demo.run_all_demos()
