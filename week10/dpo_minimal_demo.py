import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Dict
import argparse  # 将 argparse 导入到文件顶部

warnings.filterwarnings("ignore", category=FutureWarning, module="torch.cuda")
warnings.filterwarnings("ignore", message=".*NVIDIA GeForce RTX.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*generation flags are not valid and may be ignored:.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated! Use `dtype` instead!.*", category=FutureWarning)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW


QWEN_CHAT_TEMPLATE = (
    "<|im_start|>user\n{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def tokenize_batch(tokenizer: AutoTokenizer, texts: List[str], max_len: int = 512) -> Dict[str, torch.Tensor]:
    out = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=False)
    return {k: v for k, v in out.items()}


@dataclass
class PreferenceItem:
    prompt: str
    chosen: str
    rejected: str


class PreferenceDataset(Dataset):
    def __init__(self, items: List[PreferenceItem]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {"prompt": item.prompt, "chosen": item.chosen, "rejected": item.rejected}


def sequence_logprob(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, original_prompt: str, response: str, device: str) -> torch.Tensor:
    formatted_prompt = QWEN_CHAT_TEMPLATE.format(instruction=original_prompt)
    full_text = formatted_prompt + response + tokenizer.eos_token

    full_input_ids = tokenizer(full_text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prompt_input_ids = tokenizer(formatted_prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prompt_len = prompt_input_ids.size(1)

    out = model(input_ids=full_input_ids)
    logits = out.logits[:, :-1, :]
    labels = full_input_ids[:, 1:]

    start_idx = prompt_len - 1
    
    response_len = labels.size(1) - start_idx
    if response_len <= 0:
        return torch.tensor(-1e9, device=device)

    token_logits = logits[:, start_idx:, :]
    token_labels = labels[:, start_idx:]

    log_probs = torch.nn.functional.log_softmax(token_logits, dim=-1)
    sel = torch.gather(log_probs, dim=-1, index=token_labels.unsqueeze(-1)).squeeze(-1)
    return sel.sum(dim=1) / response_len # 使用 response_len 归一化，更准确


def dpo_loss(pi_logp_chosen: torch.Tensor, pi_logp_rejected: torch.Tensor,
             ref_logp_chosen: torch.Tensor, ref_logp_rejected: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor: # beta 通常也可设为较小的值，如 0.1
    diff = (pi_logp_chosen - pi_logp_rejected) - (ref_logp_chosen - ref_logp_rejected)
    # 低学习率可以保证稳定性
    return -torch.nn.functional.logsigmoid(beta * diff).mean()


def strip_qwen_think_tags(text: str) -> str:
    import re
    cleaned_text = re.sub(r'<\|im_start\|>thought\\n.*?<\|im_end\|>', '', text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<think>(.*?<\/think>)?', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = cleaned_text.replace('<|im_end|>', '')
    cleaned_text = cleaned_text.replace(QWEN_CHAT_TEMPLATE.format(instruction=''), '')
    cleaned_text = cleaned_text.strip()
    lines = [line.strip() for line in cleaned_text.split('\n') if line.strip()]
    if lines:
        return lines[-1]
    return cleaned_text


def main(args):
    """
    DPO 极简教学 Demo 的主函数。
    """
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"当前运行设备: {device.upper()}")

    # 设置随机种子
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_name = args.model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer 加载完成。pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}")

    items = [
        PreferenceItem(prompt="将'你好世界'翻译成英文", chosen="Hello World", rejected="Hi Universe"),
        PreferenceItem(prompt="把下面句子改礼貌：把会议纪要发我", chosen="请在方便时将会议纪要发送给我，谢谢。", rejected="把会议纪要发给我。"),
        PreferenceItem(prompt="一句话解释损失掩码", chosen="损失掩码让模型只在答案区域计算损失，专注学习输出。", rejected="损失掩码是一个很重要的东西。"),
        PreferenceItem(prompt="用一句话描述 DPO 的作用", chosen="DPO 直接优化模型以匹配人类偏好，无需奖励模型。", rejected="DPO 是一个复杂的强化学习算法。"),
        PreferenceItem(prompt="请用一句话概括 DPO 的优点", chosen="DPO 简化了对齐过程，训练稳定且计算效率高。", rejected="DPO 需要大量计算资源和复杂的奖励模型。"),
    ]
    dataset = PreferenceDataset(items)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print(f"\n加载模型: {model_name}...")
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    print(f"使用的数据类型: {dtype}")
    ref_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    print(f"参考模型已加载到 {device}")

    pi_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype).to(device)
    pi_model.train()
    print(f"策略模型已加载到 {device}")

    # 使用 args 中的学习率
    optim = AdamW(pi_model.parameters(), lr=args.lr)

    print(f"\n{'='*20} 开始 DPO 训练 {'='*20}")
    for epoch in range(args.epochs):
        total_loss = 0.0
        print(f"\n{'='*20} Epoch {epoch+1}/{args.epochs} {'='*20}")
        for step, batch in enumerate(loader):
            item_prompt = batch["prompt"][0]
            item_chosen = batch["chosen"][0]
            item_rejected = batch["rejected"][0]

            with torch.no_grad():
                ref_logp_c = sequence_logprob(ref_model, tokenizer, item_prompt, item_chosen, device)
                ref_logp_r = sequence_logprob(ref_model, tokenizer, item_prompt, item_rejected, device)

            optim.zero_grad()
            with torch.autocast(device_type=device, dtype=dtype):
                pi_logp_c = sequence_logprob(pi_model, tokenizer, item_prompt, item_chosen, device)
                pi_logp_r = sequence_logprob(pi_model, tokenizer, item_prompt, item_rejected, device)
                loss = dpo_loss(pi_logp_c, pi_logp_r, ref_logp_c, ref_logp_r, beta=0.1) # 使用较小的 beta
            
            print(f"    --- Debug Info (Step {step+1}) ---")
            print(f"    ref_logp_c: {ref_logp_c.item():.4f}, ref_logp_r: {ref_logp_r.item():.4f}")
            print(f"    pi_logp_c: {pi_logp_c.item():.4f}, pi_logp_r: {pi_logp_r.item():.4f}")
            pi_diff_logp = pi_logp_c - pi_logp_r
            ref_diff_logp = ref_logp_c - ref_logp_r
            print(f"    Pi Diff: {pi_diff_logp.item():.4f}, Ref Diff: {ref_diff_logp.item():.4f}")
            diff = pi_diff_logp - ref_diff_logp
            print(f"    Overall Diff: {diff.item():.4f}, Loss: {loss.item():.4f}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(pi_model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item()
            
            if (step + 1) % 1 == 0 or step == len(loader) - 1:
                current_avg_loss = total_loss / (step + 1)
                print(f"  Step {step+1}/{len(loader)}: 当前平均 DPO Loss={current_avg_loss:.4f}")

        print(f"Epoch {epoch+1} 结束: 平均 DPO Loss={total_loss/len(loader):.4f}")

    pi_model.eval()
    print(f"\n{'='*20} 开始推理演示 (策略模型 vs 参考模型) {'='*20}")
    prompts_for_inference = [
        "将'你好世界'翻译成英文", "把下面句子改礼貌：把会议纪要发我",
        "一句话解释损失掩码", "用一句话描述 DPO 的作用",
    ]
    for ptxt in prompts_for_inference:
        formatted_inference_prompt = QWEN_CHAT_TEMPLATE.format(instruction=ptxt)
        inputs = tokenizer(formatted_inference_prompt, return_tensors="pt").to(device)
        print(f"\n--- Prompt ---\n{ptxt}")
        
        # 参考模型生成
        with torch.no_grad():
            gen_ref = ref_model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        decoded_ref = tokenizer.decode(gen_ref[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"--- 参考模型 (Ref) 回复 ---\n{decoded_ref}")

        # 策略模型生成
        with torch.no_grad():
            gen_pi = pi_model.generate(**inputs, max_new_tokens=64, pad_token_id=tokenizer.eos_token_id)
        decoded_pi = tokenizer.decode(gen_pi[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"--- 策略模型 (Pi) 回复 ---\n{decoded_pi}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Minimal Demo")
    parser.add_argument("--model", default=os.environ.get("DPO_DEMO_MODEL", "Qwen/Qwen1.5-1.8B-Chat"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-6) 
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    main(args)