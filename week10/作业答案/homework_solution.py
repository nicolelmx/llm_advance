import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

class SFTDataset(Dataset):
    
    def __init__(self, data, tokenizer, max_length=512):
        """
        初始化数据集
        Args:
            data: 原始数据列表，每个元素包含instruction和output
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 预处理所有数据，提高效率
        self.processed_data = []
        for item in self.data:
            processed_item = self._process_single_item(item)
            self.processed_data.append(processed_item)
        
    def _process_single_item(self, item):
        """
        处理单个数据项
        """
        # 对话模板
        QWEN_CHAT_TEMPLATE = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # 1. 获取instruction和output
        instruction = item["instruction"]
        output = item["output"]
        
        # 2. 使用模板格式化instruction
        formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        
        # 3. 分别对格式化后的instruction和output进行分词
        prompt_ids = self.tokenizer(formatted_prompt, add_special_tokens=False)
        response_ids = self.tokenizer(output + self.tokenizer.eos_token, add_special_tokens=False)
        
        # 4. 拼接input_ids和attention_mask
        input_ids = prompt_ids["input_ids"] + response_ids["input_ids"]
        attention_mask = prompt_ids["attention_mask"] + response_ids["attention_mask"]
        
        # 5. 构造labels：instruction部分为-100，output部分为对应的token ids
        labels = [-100] * len(prompt_ids["input_ids"]) + response_ids["input_ids"]
        
        # 6. 处理长度截断 - 确保长度不超过最大长度
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]
        
        # 7. 确保所有序列长度一致
        assert len(input_ids) == len(attention_mask) == len(labels), "序列长度不一致"
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long)
        }
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        """
        return self.processed_data[idx]

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 实现训练循环
    for batch in dataloader:
        try:
            # 1. 将数据移动到指定设备
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # 2. 清零梯度
            optimizer.zero_grad()
            
            # 3. 前向传播并计算损失
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            
            # 4. 反向传播
            loss.backward()
            
            # 5. 更新参数
            optimizer.step()
            
            # 6. 累加损失
            total_loss += loss.item()
            num_batches += 1
            
        except RuntimeError as e:
            print(f"训练批次出错: {e}")
            print(f"批次形状: input_ids={batch['input_ids'].shape}, attention_mask={batch['attention_mask'].shape}")
            continue
    
    return total_loss / max(num_batches, 1)

def generate_response(model, tokenizer, instruction, device, max_length=100):
    model.eval()
    
    # 对话模板
    QWEN_CHAT_TEMPLATE = (
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    with torch.no_grad():
        # 1. 使用模板格式化instruction
        formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        
        # 2. 使用tokenizer进行分词
        inputs = tokenizer(formatted_prompt, return_tensors="pt")
        
        # 3. 将输入移动到设备上
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # 4. 使用model.generate()生成回复
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 5. 解码生成的token ids
        # 只取新生成的部分（去掉输入部分）
        new_tokens = generated_ids[0][input_ids.shape[1]:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response

# ========== 第四部分：主函数练习 ==========
def main():

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 1. 加载Qwen模型和分词器
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # 使用较小的模型进行演示
    
    try:
        print("正在加载模型和分词器...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # 使用更稳定的模型加载配置
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32,  # 使用float32避免精度问题
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # 将模型移动到设备
        model = model.to(device)
        
        # 设置pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        print("模型加载完成！")
        
    except Exception as e:
        print(f"模型加载失败: {e}")
        print("请检查网络连接或使用本地模型路径")
        return
    
    # 2. 准备训练数据
    training_data = [
        {
            "instruction": "用两句话解释什么是机器学习",
            "output": "机器学习是人工智能的一个分支，它让计算机能够从数据中自动学习规律。通过算法训练，机器可以识别模式并做出预测或决策。"
        },
        {
            "instruction": "什么是深度学习？",
            "output": "深度学习是机器学习的一个子领域，使用多层神经网络来模拟人脑的学习过程。它能够自动提取数据的特征，在图像识别、自然语言处理等领域表现优异。"
        },
        {
            "instruction": "解释一下监督学习和无监督学习的区别",
            "output": "监督学习使用带标签的训练数据，目标是学习输入和输出之间的映射关系。无监督学习则处理没有标签的数据，主要任务是发现数据中的隐藏模式或结构。"
        },
        {
            "instruction": "什么是神经网络？",
            "output": "神经网络是受生物神经网络启发的计算模型，由多个相互连接的节点（神经元）组成。每个节点接收输入信号，经过处理后传递给下一层节点，最终产生输出结果。"
        },
        {
            "instruction": "解释一下过拟合现象",
            "output": "过拟合是指模型在训练数据上表现很好，但在新数据上表现较差的现象。这通常发生在模型过于复杂或训练数据不足时，模型记住了训练数据的细节而不是学习到通用的规律。"
        }
    ]
    
    # 3. 创建SFTDataset实例
    print("正在创建数据集...")
    # 减少最大长度以避免维度不匹配问题
    dataset = SFTDataset(training_data, tokenizer, max_length=256)
    print(f"数据集大小: {len(dataset)}")
    
    # 4. 创建DataLoader
    # 使用自定义的DataCollator来处理批次数据
    data_collator = DataCollator(tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=data_collator)
    
    # 5. 设置AdamW优化器
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 6. 进行训练
    print("开始训练...")
    num_epochs = 3
    
    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, dataloader, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, 平均损失: {avg_loss:.4f}")
    
    print("训练完成！")
    
    # 7. 使用generate_response测试模型效果
    print("\n=== 测试推理效果 ===")
    test_instructions = [
        "什么是人工智能？",
        "解释一下什么是算法",
        "什么是数据科学？"
    ]
    
    for instruction in test_instructions:
        print(f"\n问题: {instruction}")
        response = generate_response(model, tokenizer, instruction, device)
        print(f"回答: {response}")

# 数据整理器类
class DataCollator:
    """自定义数据整理器，用于处理批次数据"""
    
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        # 找到批次中的最大长度
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            # 计算需要填充的长度
            pad_len = max_len - len(item["input_ids"])
            
            # 填充input_ids
            padded_input_ids = torch.cat([
                item["input_ids"], 
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
            ])
            
            # 填充attention_mask
            padded_attention_mask = torch.cat([
                item["attention_mask"], 
                torch.zeros(pad_len, dtype=torch.long)
            ])
            
            # 填充labels
            padded_labels = torch.cat([
                item["labels"], 
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
            
            input_ids.append(padded_input_ids)
            attention_mask.append(padded_attention_mask)
            labels.append(padded_labels)
        
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels)
        }

if __name__ == "__main__":
    main()
