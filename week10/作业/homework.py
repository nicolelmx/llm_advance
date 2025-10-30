import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# ========== 第一部分：数据预处理练习 ==========
class SFTDataset(Dataset):
    """
    练习1：完成SFT数据集的实现
    要求：
    1. 实现__init__方法，接收原始数据并存储
    2. 实现__len__方法
    3. 实现__getitem__方法，返回处理后的样本
    """
    
    def __init__(self, data, tokenizer, max_length=256):
        """
        初始化数据集
        Args:
            data: 原始数据列表，每个元素包含instruction和output
            tokenizer: 分词器
            max_length: 最大序列长度（建议使用256避免维度问题）
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # TODO: 在这里添加你的初始化代码
        # 提示：可以预处理数据以提高效率，或者保持简单在__getitem__中处理
        
    def __len__(self):
        """返回数据集大小"""
        # TODO: 实现这个方法
        pass
    
    def __getitem__(self, idx):
        """
        获取单个样本
        要求：
        1. 获取instruction和output
        2. 使用QWEN_CHAT_TEMPLATE格式化
        3. 分别对instruction和output进行分词
        4. 构造labels，instruction部分设为-100
        5. 处理长度截断
        """
        # 对话模板
        QWEN_CHAT_TEMPLATE = (
            "<|im_start|>user\n{instruction}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        # TODO: 实现数据预处理逻辑
        # 提示：参考以下步骤
        # 1. 获取当前样本的instruction和output
        # 2. 使用模板格式化instruction
        # 3. 分别对格式化后的instruction和output进行分词
        # 4. 拼接input_ids和attention_mask
        # 5. 构造labels：instruction部分为-100，output部分为对应的token ids
        # 6. 处理长度截断（重要：确保不超过max_length）
        # 7. 确保所有张量长度一致
        
        # 示例实现提示：
        # instruction = self.data[idx]["instruction"]
        # output = self.data[idx]["output"]
        # formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
        # prompt_ids = self.tokenizer(formatted_prompt, add_special_tokens=False)
        # response_ids = self.tokenizer(output + self.tokenizer.eos_token, add_special_tokens=False)
        # ... 继续实现
        
        return {
            "input_ids": None,  # 替换为实际值
            "attention_mask": None,  # 替换为实际值
            "labels": None  # 替换为实际值
        }

# ========== 第二部分：训练函数练习 ==========
def train_one_epoch(model, dataloader, optimizer, device):
    """
    练习2：实现一个训练轮次
    要求：
    1. 设置模型为训练模式
    2. 遍历dataloader
    3. 计算损失并反向传播
    4. 返回平均损失
    """
    model.train()
    total_loss = 0
    
    # TODO: 实现训练循环
    # 提示：
    # 1. 使用for循环遍历dataloader
    # 2. 将数据移动到指定设备
    # 3. 清零梯度
    # 4. 前向传播并计算损失
    # 5. 反向传播
    # 6. 更新参数
    # 7. 累加损失
    # 8. 添加异常处理以避免维度不匹配错误
    
    # 示例实现提示：
    # for batch in dataloader:
    #     try:
    #         input_ids = batch["input_ids"].to(device)
    #         attention_mask = batch["attention_mask"].to(device)
    #         labels = batch["labels"].to(device)
    #         optimizer.zero_grad()
    #         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    #         loss = outputs.loss
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #     except RuntimeError as e:
    #         print(f"批次处理出错: {e}")
    #         continue
    
    return total_loss / max(len(dataloader), 1)

# ========== 第三部分：推理函数练习 ==========
def generate_response(model, tokenizer, instruction, device, max_length=100):
    """
    练习3：实现模型推理函数
    要求：
    1. 使用对话模板格式化输入
    2. 进行分词
    3. 生成回复
    4. 解码并返回结果
    """
    model.eval()
    
    # 对话模板
    QWEN_CHAT_TEMPLATE = (
        "<|im_start|>user\n{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    # TODO: 实现推理逻辑
    # 提示：
    # 1. 使用模板格式化instruction
    # 2. 使用tokenizer进行分词
    # 3. 将输入移动到设备上
    # 4. 使用model.generate()生成回复
    # 5. 解码生成的token ids
    
    # 示例实现提示：
    # with torch.no_grad():
    #     formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
    #     inputs = tokenizer(formatted_prompt, return_tensors="pt")
    #     input_ids = inputs["input_ids"].to(device)
    #     attention_mask = inputs["attention_mask"].to(device)
    #     generated_ids = model.generate(
    #         input_ids=input_ids,
    #         attention_mask=attention_mask,
    #         max_new_tokens=max_length,
    #         do_sample=True,
    #         temperature=0.7,
    #         pad_token_id=tokenizer.eos_token_id
    #     )
    #     new_tokens = generated_ids[0][input_ids.shape[1]:]
    #     response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    #     return response
    
    return "请实现这个函数"

# ========== 第四部分：主函数练习 ==========
def main():
    """
    练习4：完成主函数的实现
    要求：
    1. 加载模型和分词器
    2. 准备训练数据
    3. 创建数据集和数据加载器
    4. 设置优化器
    5. 进行训练
    6. 测试推理效果
    """
    
    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # TODO: 实现主函数逻辑
    # 提示：
    # 1. 加载Qwen模型和分词器（建议使用较小的模型如Qwen2.5-0.5B-Instruct）
    # 2. 准备一些简单的训练数据（至少3个样本）
    # 3. 创建SFTDataset实例（使用max_length=256）
    # 4. 创建DataLoader
    # 5. 设置AdamW优化器
    # 6. 调用train_one_epoch进行训练
    # 7. 使用generate_response测试模型效果
    
    # 示例实现提示：
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name, 
    #     torch_dtype=torch.float32,
    #     trust_remote_code=True
    # )
    # model = model.to(device)
    # if tokenizer.pad_token is None:
    #     tokenizer.pad_token = tokenizer.eos_token
    # ... 继续实现
    
    # 示例训练数据格式
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
        }
    ]
    
    print("请完成主函数的实现...")

if __name__ == "__main__":
    main()