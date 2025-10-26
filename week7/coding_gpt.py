import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Model
from torch.optim import AdamW
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class GPTTeachingDemo:
    def __init__(self):
        """初始化GPT教学演示类"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

    def load_gpt_model(self):
        """1. 加载预训练GPT模型"""
        print("=== 1. 加载GPT模型 ===")

        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained('gpt2')

            # 设置pad token
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # 加载GPT模型
            self.gpt_model = GPT2LMHeadModel.from_pretrained('gpt2').to(self.device)
            self.gpt_base = GPT2Model.from_pretrained('gpt2').to(self.device)

            print("✓ GPT模型加载完成")
            print(f"模型参数量: {sum(p.numel() for p in self.gpt_model.parameters()):,}")
            return True

        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("💡 请检查网络连接或下载模型文件到本地")
            print("   离线模式: 设置环境变量 HF_HUB_OFFLINE=1")
            print("   或手动下载模型到: ~/.cache/huggingface/hub/")
            return False

    def demonstrate_gpt_tokenization(self):
        """2. 演示GPT分词"""
        print("\n=== 2. GPT分词演示 ===")

        text = "Natural language processing is revolutionizing AI"

        # 基本分词
        tokens = self.tokenizer.tokenize(text)
        print(f"原始文本: {text}")
        print(f"分词结果: {tokens}")

        # 转换为ID
        input_ids = self.tokenizer.encode(text, add_special_tokens=True)
        print(f"输入ID: {input_ids}")
        print(f"特殊token: [EOS]={self.tokenizer.eos_token_id}")

    def demonstrate_gpt_forward(self):
        """3. 演示GPT前向传播"""
        print("\n=== 3. GPT前向传播演示 ===")

        text = "The future of AI is"
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.gpt_model(**inputs, labels=inputs['input_ids'])

        print(f"输入文本: {text}")
        print(f"输出logits形状: {outputs.logits.shape}")
        print(f"损失值: {outputs.loss:.4f}")

    def demonstrate_text_generation(self):
        """4. 演示文本生成能力"""
        print("\n=== 4. GPT文本生成演示 ===")

        # 基础文本生成
        prompt = "The benefits of artificial intelligence include"
        print(f"提示文本: {prompt}")

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            # 生成文本
            generated_outputs = self.gpt_model.generate(
                inputs['input_ids'],
                max_length=50,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(generated_outputs[0], skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
        print()

        # 演示不同参数的效果
        print("=== 不同生成参数对比 ===")

        # 高创造性生成 (高温度)
        creative_outputs = self.gpt_model.generate(
            inputs['input_ids'],
            max_length=30,
            temperature=1.2,
            do_sample=True,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        creative_text = self.tokenizer.decode(creative_outputs[0], skip_special_tokens=True)
        print(f"高创造性: {creative_text}")

        # 保守生成 (低温度)
        conservative_outputs = self.gpt_model.generate(
            inputs['input_ids'],
            max_length=30,
            temperature=0.3,
            do_sample=True,
            top_k=50,
            pad_token_id=self.tokenizer.eos_token_id
        )
        conservative_text = self.tokenizer.decode(conservative_outputs[0], skip_special_tokens=True)
        print(f"保守生成: {conservative_text}")

    def demonstrate_autoregressive_modeling(self):
        """5. 演示自回归语言建模"""
        print("\n=== 5. 自回归语言建模演示 ===")

        text = "The weather today is"
        print(f"输入序列: {text}")

        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)

        with torch.no_grad():
            outputs = self.gpt_model(**inputs)
            next_token_logits = outputs.logits[:, -1, :]

        # 获取Top 5预测
        top_k = 5
        top_tokens = torch.topk(next_token_logits, top_k, dim=-1)

        print("预测下一个token (Top 5):")
        for i in range(top_k):
            token_id = top_tokens.indices[0, i].item()
            token = self.tokenizer.decode(token_id)
            prob = torch.softmax(next_token_logits, dim=-1)[0, token_id].item()
            print(f"  {token}: {prob:.4f}")

    def create_generation_dataset(self):
        """创建文本生成数据集"""
        # 模拟故事续写数据集
        prompts = [
            "Once upon a time, in a magical forest,",
            "The scientist discovered a new element that",
            "In the year 2050, artificial intelligence",
            "The young adventurer found an ancient map leading to"
        ]
        return prompts

    def fine_tune_gpt_generator(self):
        """6. GPT微调实战：故事续写"""
        print("\n=== 6. GPT微调实战：故事续写 ===")

        # 准备数据
        prompts = self.create_generation_dataset()

        # 训练参数
        optimizer = AdamW(self.gpt_model.parameters(), lr=5e-5)
        num_epochs = 2

        print("开始微调训练...")

        for epoch in range(num_epochs):
            total_loss = 0

            for prompt in prompts:
                # 编码输入
                inputs = self.tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(self.device)

                # 创建标签 (自回归任务)
                labels = inputs['input_ids'].clone()

                # 前向传播
                outputs = self.gpt_model(**inputs, labels=labels)
                loss = outputs.loss

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(prompts)
            print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        print("✓ 微调完成")

        # 测试微调效果
        test_prompt = "In a distant galaxy,"
        test_inputs = self.tokenizer(test_prompt, return_tensors='pt').to(self.device)

        with torch.no_grad():
            test_outputs = self.gpt_model.generate(
                test_inputs['input_ids'],
                max_length=40,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_story = self.tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        print(f"\n微调后生成故事:\n{generated_story}")

    def demonstrate_gpt_evolution(self):
        """7. GPT技术演进演示"""
        print("\n=== 7. GPT技术演进 ===")

        evolution = {
            "GPT-1": {
                "参数量": "1.17亿",
                "特点": "首次大规模预训练",
                "性能": "基础语言建模"
            },
            "GPT-2": {
                "参数量": "15亿",
                "特点": "零样本学习能力",
                "性能": "多任务适应性强"
            },
            "GPT-3": {
                "参数量": "1750亿",
                "特点": "少样本和零样本学习",
                "性能": "接近人类水平"
            },
            "GPT-4": {
                "参数量": "未知",
                "特点": "多模态能力",
                "性能": "更强的推理能力"
            }
        }

        print("GPT系列模型演进:")
        for version, info in evolution.items():
            print(f"\n{version}:")
            for key, value in info.items():
                print(f"  {key}: {value}")

    def compare_gpt_vs_bert(self):
        """8. GPT与BERT对比分析"""
        print("\n=== 8. GPT vs BERT 对比分析 ===")

        comparison = {
            "架构设计": {
                "GPT": "单向Transformer解码器（自回归）",
                "BERT": "双向Transformer编码器（自编码）"
            },
            "注意力机制": {
                "GPT": "掩码自注意力（因果掩码）",
                "BERT": "全词注意力（无掩码）"
            },
            "预训练任务": {
                "GPT": "自回归语言建模（预测下一词）",
                "BERT": "掩码语言建模（MLM）+下一句预测（NSP）"
            },
            "核心优势": {
                "GPT": "文本生成、创意写作、对话系统",
                "BERT": "文本理解、分类、问答系统"
            },
            "应用场景": {
                "GPT": "内容创作（采纳率72%）、代码生成",
                "BERT": "文本分类（准确率94.9%）、NER（F1值96.6%）"
            },
            "参数规模": {
                "GPT": "GPT-3达1750亿参数",
                "BERT": "Base版1.1亿参数"
            }
        }

        print("GPT与BERT核心对比:")
        for aspect, models in comparison.items():
            print(f"\n{aspect}:")
            print(f"  GPT: {models['GPT']}")
            print(f"  BERT: {models['BERT']}")

    def demonstrate_generation_strategies(self):
        """9. 文本生成策略演示"""
        print("\n=== 9. 文本生成策略演示 ===")

        prompt = "Machine learning is"

        strategies = {
            "贪婪解码": {"do_sample": False, "temperature": 1.0},
            "随机采样": {"do_sample": True, "temperature": 1.0, "top_k": 50},
            "Top-p采样": {"do_sample": True, "temperature": 0.8, "top_p": 0.9},
            "Top-k采样": {"do_sample": True, "temperature": 0.8, "top_k": 40}
        }

        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)

        print(f"提示: {prompt}")
        print("\n不同生成策略结果:")

        for strategy_name, params in strategies.items():
            with torch.no_grad():
                outputs = self.gpt_model.generate(
                    inputs['input_ids'],
                    max_length=20,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **params
                )

            generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"\n{strategy_name}: {generated[len(prompt):].strip()}")

    def run_all_demos(self):
        """运行所有GPT教学演示"""
        print("🚀 开始GPT教学演示\n")

        if not self.load_gpt_model():
            print("\n⚠️  由于模型加载失败，跳过需要模型的演示")
            print("📖 您可以查看代码注释了解GPT的工作原理")
            return

        self.demonstrate_gpt_tokenization()
        self.demonstrate_gpt_forward()
        self.demonstrate_text_generation()
        self.demonstrate_autoregressive_modeling()
        self.fine_tune_gpt_generator()
        self.demonstrate_gpt_evolution()
        self.compare_gpt_vs_bert()
        self.demonstrate_generation_strategies()

        print("\n🎉 GPT教学演示完成！")

# 主程序
if __name__ == "__main__":
    demo = GPTTeachingDemo()
    demo.run_all_demos()