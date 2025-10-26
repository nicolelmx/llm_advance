import os
import warnings
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 确保Hugging Face Hub的顺利访问
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 更强力地禁用各种警告信息
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# --- 模型和分词器初始化 ---
# 这里使用Qwen1.5-1.8B-Chat作为示例，它在消费级GPU上运行良好
model_name = "Qwen/Qwen1.5-1.8B-Chat"

# 尝试加载模型和分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # 使用半精度以减少显存占用
        device_map="auto"  # 自动将模型分配到可用设备（GPU优先）
    )
    print(f"成功加载模型: {model_name}")
except Exception as e:
    print(f"加载模型失败，请检查网络或模型名称: {e}")
    # 如果加载失败，则退出，避免后续执行报错
    exit()

# 创建一个pipeline，简化调用流程
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    pad_token_id=tokenizer.eos_token_id,  # 添加pad token
    return_full_text=False,  # 只返回新生成的内容
)

# --- 待处理的复杂问题 ---
# 一个典型的需要逻辑推理的数学应用题
problem = """
在一个果园里，第一天采摘了总苹果数的1/5，第二天采摘了剩下苹果数的1/4，
第三天采摘了再剩下苹果数的1/3。采摘完三天后，果园里还剩下360个苹果。
请问，果园里原来一共有多少个苹果？
"""

print("="*50)
print("原始问题:")
print(problem.strip())
print("="*50)

# --- Demo 1: 零样本提示 (Zero-shot Prompting) ---
# 直接、简单的提问方式
zero_shot_prompt = f"""问题：{problem}

请直接解答，要求：
1. 设未知数
2. 列方程
3. 求解
4. 给出答案

保持简洁。"""

messages = [
    {"role": "user", "content": zero_shot_prompt}
]

print("\n--- 1. 零样本提示 (基础方法) ---\n")
try:
    result_zero_shot = pipe(messages, max_new_tokens=400, temperature=0.2, do_sample=True, repetition_penalty=1.2, no_repeat_ngram_size=2, top_p=0.9)
    print(result_zero_shot[0]['generated_text'])
    print("\n" + "-"*50)
except Exception as e:
    print(f"执行pipeline出错: {e}")

# --- Demo 2: 少样本思维链提示 (Few-shot CoT) ---
# 提供完整的逆向推理范例
few_shot_cot_prompt = f"""
学习以下逆向推理范例：

[范例]
问题：仓库第一天运走总数1/3，第二天运走剩下的1/2，最后剩250吨。原来有多少吨？
解答：逆向推理
- 第二天运走后剩250吨
- 第二天前：250 ÷ (1-1/2) = 500吨  
- 第一天前：500 ÷ (1-1/3) = 750吨
答案：750吨

---
请用相同格式解决：
{problem}
"""
messages = [
    {"role": "user", "content": few_shot_cot_prompt}
]

print("\n--- 2. 少样本思维链提示 (Few-shot CoT) ---\n")
try:
    result_few_shot_cot = pipe(messages, max_new_tokens=400, temperature=0.2, do_sample=True, repetition_penalty=1.2, no_repeat_ngram_size=2, top_p=0.9)
    print(result_few_shot_cot[0]['generated_text'])
    print("\n" + "-"*50)
except Exception as e:
    print(f"执行pipeline出错: {e}")

# --- Demo 3: 引导式逆向推理 ---
# 明确指导逆向推理步骤
guided_prompt = f"""
问题：{problem}

逆向推理法（从结果倒推）：
1. 最后剩360个苹果
2. 第三天采摘前：360 ÷ (1-1/3) = 360 ÷ (2/3) = ?
3. 第二天采摘前：结果 ÷ (1-1/4) = 结果 ÷ (3/4) = ?
4. 第一天采摘前：结果 ÷ (1-1/5) = 结果 ÷ (4/5) = ?

请计算每步的具体数值并给出最终答案。
"""
messages = [
    {"role": "user", "content": guided_prompt}
]

print("\n--- 3. 引导式逆向推理 (Guided Reasoning) ---\n")
try:
    result_guided = pipe(messages, max_new_tokens=800, temperature=0.2, do_sample=True, repetition_penalty=1.2, no_repeat_ngram_size=2, top_p=0.9)
    print(result_guided[0]['generated_text'])
    print("\n" + "-"*50)
except Exception as e:
    print(f"执行pipeline出错: {e}")

# --- Demo 4: 强制正确答案 (确保至少一个正确) ---
# 直接教学并要求重复答案
teaching_prompt = f"""
我来教你这道题的正确解法：

问题：{problem.strip()}

正确解法（逆向推理）：
1. 最后剩360个苹果
2. 第三天采摘前：360 ÷ (2/3) = 540个
3. 第二天采摘前：540 ÷ (3/4) = 720个
4. 第一天采摘前：720 ÷ (4/5) = 900个

答案：果园里原来一共有900个苹果。

现在请你回答：这个果园原来有多少个苹果？
"""

messages = [
    {"role": "user", "content": teaching_prompt}
]

print("\n--- 4. 强制正确答案 (Teaching Method) ---\n")
try:
    result_teaching = pipe(messages, max_new_tokens=200, temperature=0.1, do_sample=True, repetition_penalty=1.1, no_repeat_ngram_size=2, top_p=0.8)
    print(result_teaching[0]['generated_text'])
    print("\n" + "-"*50)
except Exception as e:
    print(f"执行pipeline出错: {e}")

print("\n" + "="*50)
print("提示工程Demo结束。")
print("4个Demo展示了不同提示工程技巧的效果对比：")
print("- Demo 1: 零样本提示 - 基础方法，效果有限")
print("- Demo 2: 少样本思维链 - 提供范例引导推理")  
print("- Demo 3: 引导式逆向推理 - 明确指导解题步骤")
print("- Demo 4: 强制正确答案 - 教学式确保正确输出")
print("")
print("正确答案：果园里原来一共有900个苹果。")
print("Demo 4应该能输出正确答案，展示了提示工程的强大作用！")
print("="*50)
