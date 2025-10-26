import math
import argparse
import random
import numpy as np
import os
import warnings # 导入 warnings 模块
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime # 引入 datetime 用于生成时间戳

# -------------------- 警告抑制 --------------------
# 抑制 pynvml 弃用警告
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
# 抑制 PyTorch CUDA 能力不匹配警告
warnings.filterwarnings("ignore", message="NVIDIA GeForce RTX.* is not compatible with the current PyTorch installation.", category=UserWarning)
# 抑制 Flash Attention 未编译警告
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.", category=UserWarning)
# 警告过滤规则，以消除模型生成参数无效的警告
warnings.filterwarnings("ignore", category=UserWarning, message=".*generation flags are not valid and may be ignored:.*")

# 禁用 TensorFlow oneDNN 提示（如果安装了 TensorFlow）
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# --------------------------------------------------

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from torch.optim import AdamW


QWEN_CHAT_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def build_sample(
    instruction: str,
    answer: str,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
) -> Dict[str, torch.Tensor]:
    """
    根据指令和答案构建单个训练样本的 token ID 和标签。
    会根据 `QWEN_CHAT_TEMPLATE` 格式化 prompt，并对 answer 区域进行损失掩码（标签为 -100）。

    Args:
        instruction (str): 用户指令文本。
        answer (str): 模型的期望答案文本。
        tokenizer (AutoTokenizer): 用于文本编码的 tokenizer 实例。
        max_length (int): 样本的最大序列长度，超出部分将被截断。

    Returns:
        Dict[str, torch.Tensor]: 包含 'input_ids', 'attention_mask' 和 'labels' 的字典。
                                 'labels' 中 prompt 部分为 -100，answer 部分为真实 token ID。
    """
    # 使用 Qwen 的对话模板格式化输入
    prompt = QWEN_CHAT_TEMPLATE.format(instruction=instruction)
    # 训练时，在 answer 后添加 eos_token，明确告知模型回答结束
    # Qwen 的 eos_token 即为 <|im_end|>
    prompt_ids = tokenizer(prompt, add_special_tokens=False)
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)

    input_ids = prompt_ids["input_ids"] + answer_ids["input_ids"]
    attn_mask = prompt_ids["attention_mask"] + answer_ids["attention_mask"]

    # 创建损失掩码。仅对 answer 区域计算损失，prompt 区域的 token 标签设置为 -100。
    # 这样模型在训练时只会学习生成答案，而不会尝试复述或预测 prompt 部分。
    labels = [-100] * len(prompt_ids["input_ids"]) + answer_ids["input_ids"]

    # 截断
    input_ids = input_ids[:max_length]
    attn_mask = attn_mask[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class InMemorySFTDataset(Dataset):
    """
    内存中的 SFT 数据集，用于存储和访问预处理后的训练样本。
    每个样本都是通过 `build_sample` 函数构建的 token ID 序列及其对应的损失掩码标签。

    Args:
        rows (List[Dict[str, str]]): 原始的指令-答案对列表。
        tokenizer (AutoTokenizer): 用于处理文本的 tokenizer 实例。
        max_length (int): 样本的最大序列长度。
    """
    def __init__(self, rows: List[Dict[str, str]], tokenizer: AutoTokenizer, max_length: int = 256):
        self.examples = [build_sample(r["instruction"], r["answer"], tokenizer, max_length) for r in rows]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


@dataclass
class Collator:
    """
    数据整理器，用于将批次中的样本填充（padding）到相同的长度。
    填充操作应用于 input_ids, attention_mask 和 labels。
    input_ids 和 labels 用 pad_token_id 填充，attention_mask 用 0 填充。

    Args:
        pad_token_id (int): tokenizer 的填充 token ID。
    """
    pad_token_id: int

    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        max_len = max(x["input_ids"].size(0) for x in batch)
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad_len = max_len - x["input_ids"].size(0)
            input_ids.append(torch.nn.functional.pad(x["input_ids"], (0, pad_len), value=self.pad_token_id))
            attn_mask.append(torch.nn.functional.pad(x["attention_mask"], (0, pad_len), value=0))
            labels.append(torch.nn.functional.pad(x["labels"], (0, pad_len), value=-100))
        return {
            "input_ids": torch.stack(input_ids),  # 批次中的所有 input_ids 堆叠成一个张量
            "attention_mask": torch.stack(attn_mask),  # 批次中的所有 attention_mask 堆叠成一个张量
            "labels": torch.stack(labels),  # 批次中的所有 labels 堆叠成一个张量
        }


def main():
    """
    SFT 损失掩码教学 Demo 的主函数。
    负责解析命令行参数、设置随机种子、加载模型和 tokenizer、准备数据集、
    执行训练循环、并在训练后进行少样本推理演示。
    """
    parser = argparse.ArgumentParser(description="SFT Loss Masking Demo")
    # 修改为本地 Qwen 0.6B 模型路径（设置环境变量 SFT_DEMO_MODEL）
    parser.add_argument("--model", default=os.environ.get("SFT_DEMO_MODEL", "week9/项目实战/models/Qwen/Qwen3-0.6B"))
    parser.add_argument("--epochs", type=int, default=3) # 对于大模型，先用较少 Epoch 进行快速验证
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512) # 增加最大序列长度以适应Qwen模型
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 设置随机种子，以确保实验的可复现性
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前运行设备: {device.upper()}")

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    # Qwen tokenizer 的 eos_token 就是其 <|im_end|>，可直接作为 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer 加载完成。pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

    # 定义用于训练的少量教学样本（指令-答案对）
    rows = [
        {"instruction": "用两句话解释什么是监督微调（SFT）", "answer": "SFT 是在指令-答案对上训练模型的过程。它让模型学会遵循人类指令并给出符合约束的回复。"},
        {"instruction": "把下面句子改写得更礼貌：给我周一的会议纪要", "answer": "请在方便的时候，帮我整理并发送一下周一的会议纪要，谢谢。"},
        {"instruction": "将'你好世界'翻译成英文", "answer": "Hello World"},
        {"instruction": "请将这句话缩写到不超过15个字：我今天需要完成报告的初稿并发送邮件", "answer": "今日完成报告初稿并邮件发送。"},
        {"instruction": "把口语句改成书面语：你把表格发我一下", "answer": "请将表格发送给我。"},
        {"instruction": "将这段回复调整为更正式的语气：好的，收到啦！", "answer": "好的，已收到。"},
        {"instruction": "把下面两句话合并为一句并保持礼貌：能否共享文档；我需要查看历史版本", "answer": "能否请您共享文档，我需要查看历史版本。"},
        {"instruction": "改写为更简洁：本周请完成需求评审并同步会议纪要", "answer": "本周请完成评审并同步纪要。"},
        {"instruction": "给出一个更自然的英文问候：早上好，今天的会议几点开始？", "answer": "Good morning! What time does today's meeting start?"},
        {"instruction": "把这句英文翻译成自然中文：Please let me know if you have any questions.", "answer": "如有任何问题，请随时告知。"},
        {"instruction": "润色：感谢你的帮助，我会尽快反馈结果", "answer": "感谢您的帮助，我会尽快反馈结果。"},
        # 强化“损失掩码/只在答案区计损失”的同域样本（多样化表述）
        {"instruction": "一句话解释损失掩码是什么", "answer": "损失掩码让训练只关注答案区域，避免把精力浪费在题目复述上。"},
        {"instruction": "为什么在 SFT 中不对题目部分计算损失？", "answer": "因为只关心答案质量，题目部分不计分能集中学习输出能力。"}, # 损失掩码的训练样本
        {"instruction": "用一句话概括 Loss Masking 的作用", "answer": "通过掩码只对答案打分，使模型专注学习解题而非复述题目。"},   # 损失掩码的训练样本
        {"instruction": "为什么要只在答案区计算损失？", "answer": "只在答案区计损失能提升训练效率并显著改善回答质量。"},   # 损失掩码的训练样本
        {"instruction": "损失掩码在训练中起什么效果？", "answer": "它抑制题目复述的梯度，让模型把能力集中到正确回答上。"},     # 损失掩码的训练样本
        # 进一步增加与损失掩码相关的训练样本
        {"instruction": "损失掩码对 SFT 的益处是什么？", "answer": "损失掩码能让模型更专注于生成高质量的回答，提高训练效率。"},
        {"instruction": "简述损失掩码的原理", "answer": "损失掩码通过将 prompt 区域的标签设置为 -100，使其在损失计算中被忽略。"},
        {"instruction": "损失掩码是否适用于所有任务？", "answer": "主要适用于 SFT 等指令遵循任务，有助于模型区分输入和期望输出。"},
        {"instruction": "损失掩码如何提高训练效率？", "answer": "通过避免在非关键部分计算损失，从而加速收敛并减少计算资源消耗。"},
        {"instruction": "在多轮对话中，损失掩码有什么特殊作用？", "answer": "在多轮对话中，损失掩码可以确保模型只对当前轮次的新回复进行学习和优化。"},
        {"instruction": "损失掩码与自回归训练有什么关系？", "answer": "在自回归训练中，损失掩码通常用于确保模型仅对目标序列的生成部分进行梯度更新。"},
        # 增加通用“一句话回答”训练样本，强化模型对简洁回答格式的学习
        {"instruction": "用一句话概括夏天最常见的活动", "answer": "夏天常见活动包括游泳、烧烤和海滩度假。"},
        {"instruction": "请用一句话描述人工智能", "answer": "人工智能是使机器模仿人类智能进行感知、推理、学习和解决问题的技术。"},
        {"instruction": "用一句话说明为什么学习编程很重要", "answer": "学习编程能提升逻辑思维，打开职业机会，并赋能创新。"},
    ]

    dataset = InMemorySFTDataset(rows, tokenizer, max_length=args.max_length)
    collator = Collator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    print(f"\n加载模型: {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32)

    model.to(device)
    model.train() # 将模型设置为训练模式
    print(f"模型已加载到 {device}, 参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M")

    optim = AdamW(model.parameters(), lr=args.lr)

    print(f"\n{'='*20} 开始 SFT 训练 {'='*20}")
    for epoch in range(args.epochs):
        total_loss = 0.0 # 每个 epoch 的总损失
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        for step, batch in enumerate(loader):
            # 将 batch 移动到设备
            batch = {k: v.to(device) for k, v in batch.items()}
            optim.zero_grad()
            
            with torch.autocast(device_type=device, dtype=torch.bfloat16 if device == "cuda" else torch.float32):
                out = model(**batch)
                loss = out.loss
            
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()

            if (step + 1) % 10 == 0 or step == len(loader) - 1: 
                current_avg_loss = total_loss / (step + 1)
                print(f"  Step {step+1}/{len(loader)}: 当前平均损失={current_avg_loss:.4f}")

        avg_loss = total_loss / len(loader) # 计算平均损失
        try:
            ppl = math.exp(avg_loss) # 计算困惑度 (Perplexity)
        except OverflowError: # 处理损失过大导致 ppl 溢出的情况
            ppl = float('inf')
        print(f"Epoch {epoch+1} 结束: 平均损失={avg_loss:.4f}, 困惑度 (PPL)={ppl:.2f}")

    output_dir = f"./sft_model_checkpoint_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\n模型已保存到: {output_dir}")

    model.eval()
    with torch.no_grad():
        print(f"\n{'='*20} 开始推理演示 {'='*20}")

        example_q = "为什么在 SFT 中不对题目部分计算损失？" # 示例问题
        example_a = "因为只关心答案质量，题目部分不计分能集中学习输出能力。" # 示例答案
        target_q = "请用一句话说明损失掩码的作用" # 目标问题

        few_shot_prompt = (
            "<|im_start|>user\n" + example_q + "<|im_end|>\n" +
            "<|im_start|>assistant\n" + example_a + tokenizer.eos_token + "\n" +
            "<|im_start|>user\n" + target_q + "<|im_end|>\n" +
            "<|im_start|>assistant\n"
        )

        prompt = few_shot_prompt
        print("\n--- 推理 Prompt ---")
        # 打印时，为了可读性，将特殊 token <|im_end|> 替换为 <|eos|>
        print(prompt.replace(tokenizer.eos_token, "<|eos|>")) 

        # 将输入 prompt 编码并移动到指定设备
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
 
        generated_tokens = model.generate(
            **inputs,
            max_new_tokens=64,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=False
        )
        

        new_generated_tokens = generated_tokens[:, inputs.input_ids.shape[1]:]
        
        # 解码并打印生成的答案
        decoded_answer = tokenizer.decode(new_generated_tokens[0], skip_special_tokens=True).strip()
        print("\n--- 模型生成的答案 ---")
        print(decoded_answer)


if __name__ == "__main__":
    main()