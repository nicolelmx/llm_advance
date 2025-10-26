import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BertModel, GPT2Model
from transformers import BertForSequenceClassification, GPT2LMHeadModel
import warnings
warnings.filterwarnings('ignore')

class BERTGPTComparison:
    def __init__(self):
        """初始化对比分析类"""
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def load_models(self):
        """加载BERT和GPT模型"""
        print("=== 加载模型进行对比 ===")

        try:
            # BERT模型
            self.bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
            self.bert_model = BertModel.from_pretrained('bert-base-chinese').to(self.device)
            self.bert_classifier = BertForSequenceClassification.from_pretrained(
                'bert-base-chinese', num_labels=2
            ).to(self.device)

            # GPT模型
            self.gpt_tokenizer = AutoTokenizer.from_pretrained('gpt2')
            self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token
            self.gpt_model = GPT2Model.from_pretrained('gpt2').to(self.device)
            self.gpt_generator = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)

            print("✓ 模型加载完成")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("💡 请检查网络连接或下载模型文件到本地")
            print("   离线模式: 设置环境变量 HF_HUB_OFFLINE=1")
            print("   或手动下载模型到: ~/.cache/huggingface/hub/")
            return False

    def compare_architectures(self):
        """1. 架构对比分析"""
        print("\n=== 1. 架构对比分析 ===")

        architecture_comparison = {
            "核心架构": {
                "BERT": "双向Transformer编码器",
                "GPT": "单向Transformer解码器"
            },
            "注意力机制": {
                "BERT": "全词注意力（无掩码）",
                "GPT": "因果掩码自注意力"
            },
            "输入处理": {
                "BERT": "[CLS] + 句子对",
                "GPT": "序列自回归"
            },
            "输出方式": {
                "BERT": "上下文表征向量",
                "GPT": "下一个token预测"
            }
        }

        print("架构对比:")
        for aspect, models in architecture_comparison.items():
            print(f"\n{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def compare_pretraining_tasks(self):
        """2. 预训练任务对比"""
        print("\n=== 2. 预训练任务对比 ===")

        task_comparison = {
            "主要任务": {
                "BERT": "MLM（掩码语言建模）+ NSP（下一句预测）",
                "GPT": "自回归语言建模（预测下一词）"
            },
            "训练目标": {
                "BERT": "学习双向上下文表征",
                "GPT": "学习生成概率分布"
            },
            "数据利用": {
                "BERT": "利用左右上下文",
                "GPT": "利用前序上下文"
            },
            "泛化能力": {
                "BERT": "强理解能力",
                "GPT": "强生成能力"
            }
        }

        print("预训练任务对比:")
        for aspect, models in task_comparison.items():
            print(f"\n{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def demonstrate_bert_bidirectional(self):
        """3. BERT双向性演示"""
        print("\n=== 3. BERT双向性演示 ===")

        text = "自然语言处理[NLP]是人工智能的重要分支"
        masked_text = "自然语言处理[MASK]是人工智能的重要分支"




        print(f"原始文本: {text}")
        print(f"掩码文本: {masked_text}")
        # 分词
        print(self.bert_tokenizer(masked_text))
        inputs = self.bert_tokenizer(masked_text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.bert_classifier(**inputs)  # 使用分类器演示
            logits = outputs.logits

        print("BERT能够利用左右上下文进行预测")
        print("✓ 双向注意力机制示例")

    def demonstrate_gpt_unidirectional(self):
        """4. GPT单向性演示"""
        print("\n=== 4. GPT单向性演示 ===")

        text = "The weather today is"
        print(f"输入序列: {text}")

        inputs = self.gpt_tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.gpt_generator(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        # Top 3预测
        top_k = 3
        top_tokens = torch.topk(next_token_logits, top_k, dim=-1)

        print("GPT只能利用前序上下文预测:")
        for i in range(top_k):
            token_id = top_tokens.indices[0, i].item()
            token = self.gpt_tokenizer.decode(token_id)
            prob = torch.softmax(next_token_logits, dim=-1)[0, token_id].item()
            print(f"  {token}: {prob:.4f}")

    def compare_fine_tuning(self):
        """5. 微调策略对比"""
        print("\n=== 5. 微调策略对比 ===")

        finetune_comparison = {
            "适用场景": {
                "BERT": "分类、问答、NER等理解任务",
                "GPT": "生成、对话、翻译等生成任务"
            },
            "参数效率": {
                "BERT": "新增少量参数（分类头）",
                "GPT": "通常全参数微调"
            },
            "数据需求": {
                "BERT": "需要标注数据",
                "GPT": "可使用无监督数据"
            },
            "训练目标": {
                "BERT": "最小化交叉熵损失",
                "GPT": "最小化语言建模损失"
            }
        }

        print("微调策略对比:")
        for aspect, models in finetune_comparison.items():
            print(f"\n{aspect}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def compare_performance_metrics(self):
        """6. 性能指标对比"""
        print("\n=== 6. 性能指标对比 ===")

        performance_data = {
            "GLUE基准": {
                "BERT_Base": "87.1",
                "GPT": "较弱"
            },
            "生成连贯性": {
                "BERT": "较弱",
                "GPT": "92%"
            },
            "SQuAD_F1": {
                "BERT": "93.2",
                "GPT": "较弱"
            },
            "参数量级": {
                "BERT_Base": "1.1亿",
                "GPT_3": "1750亿"
            }
        }

        print("性能指标对比:")
        print("指标\t\tBERT\t\tGPT")
        print("-" * 40)
        for metric, scores in performance_data.items():
            bert_score = scores.get('BERT_Base', scores.get('BERT', 'N/A'))
            gpt_score = scores.get('GPT_3', scores.get('GPT', 'N/A'))
            print(f"{metric:<12}\t{bert_score:<12}\t{gpt_score}")

    def compare_applications(self):
        """7. 应用场景对比"""
        print("\n=== 7. 应用场景对比 ===")

        applications = {
            "文本分类": {
                "BERT": "★★★★★ (IMDb 94.9%)",
                "GPT": "★★★☆☆"
            },
            "问答系统": {
                "BERT": "★★★★★ (SQuAD F1 93.2)",
                "GPT": "★★★☆☆"
            },
            "文本生成": {
                "BERT": "★★☆☆☆",
                "GPT": "★★★★★ (采纳率72%)"
            },
            "代码生成": {
                "BERT": "★★★☆☆",
                "GPT": "★★★★☆"
            },
            "对话系统": {
                "BERT": "★★★☆☆",
                "GPT": "★★★★★"
            },
            "命名实体识别": {
                "BERT": "★★★★★ (F1 96.6%)",
                "GPT": "★★☆☆☆"
            }
        }

        print("应用场景适用性:")
        for task, models in applications.items():
            print(f"\n{task}:")
            print(f"  BERT: {models['BERT']}")
            print(f"  GPT: {models['GPT']}")

    def demonstrate_joint_usage(self):
        """8. 联合使用演示"""
        print("\n=== 8. BERT与GPT联合使用 ===")

        print("BERT + GPT 联合应用模式:")

        joint_usage = {
            "1. BERT理解 + GPT生成": "先用BERT分析用户意图，再用GPT生成响应",
            "2. GPT生成 + BERT校验": "GPT生成内容后用BERT进行质量评估",
            "3. 多任务学习": "同一个架构同时学习理解和生成任务",
            "4. 知识增强": "BERT提供知识 grounding，GPT进行创意生成"
        }

        for mode, description in joint_usage.items():
            print(f"{mode}")
            print(f"   {description}")

    def create_decision_tree(self):
        """9. 任务选择决策树"""
        print("\n=== 9. 任务选择决策树 ===")

        decision_tree = """
根据任务特点选择合适的模型：

┌─────────────────────────────────────┐
│          你的NLP任务是什么？          │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
   ┌────▼────┐        ┌────▼────┐
   │理解任务 │        │生成任务 │
   │(分类/问答│        │(写作/对话│
   │ /NER)    │        │ /翻译)   │
   └────┬────┘        └────┬────┘
        │                   │
   ┌────▼────┐        ┌────▼────┐
   │ 选择BERT │        │ 选择GPT  │
   │ (双向优  │        │ (生成优  │
   │ 势明显)  │        │ 势明显)  │
   └─────────┘        └─────────┘
        """

        print(decision_tree)

    def run_comparison_analysis(self):
        """运行完整的对比分析"""
        print("🚀 开始BERT与GPT对比分析\n")

        if not self.load_models():
            print("\n⚠️  由于模型加载失败，跳过需要模型的演示")
            print("📖 展示理论对比分析:")

        self.compare_architectures()
        self.compare_pretraining_tasks()
        self.demonstrate_bert_bidirectional()
        # self.compare_fine_tuning()
        # self.compare_performance_metrics()
        # self.compare_applications()
        # self.demonstrate_joint_usage()
        # self.create_decision_tree()

        # print("\n🎉 BERT与GPT对比分析完成！")
        #
        # # 总结
        # print("\n" + "="*50)
        # print("📊 对比总结:")
        # print("• BERT: 擅长理解任务，适合分类、问答、NER等")
        # print("• GPT: 擅长生成任务，适合写作、对话、代码生成等")
        # print("• 实际应用中可根据具体需求选择或联合使用")
        # print("="*50)

# 主程序
if __name__ == "__main__":
    comparison = BERTGPTComparison()
    comparison.run_comparison_analysis()
