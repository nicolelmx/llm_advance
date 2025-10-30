import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def prompt_engineering_demo():
    """提示工程实战演示 - 数学推理问题"""
    
    print("=== 提示工程实战演示 ===")
    
    # --- 环境设置 ---
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["WANDB_DISABLED"] = "true"
    
    # --- 1. 模型加载 ---
    print("1. 加载模型...")
    model_name = "Qwen/Qwen1.5-1.8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 创建pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )
    
    # --- 2. 问题定义 ---
    problem = """
问题：一个班级有30名学生，其中60%是女生。如果班级中40%的学生参加了数学竞赛，且参加竞赛的学生中男女比例是2:3，那么有多少名女生参加了数学竞赛？
"""
    
    print("数学问题:")
    print(problem)
    
    # --- 3. 零样本提示 ---
    print("\n" + "="*60)
    print("方法1: 零样本提示 (Zero-shot)")
    print("="*60)
    
    zero_shot_prompt = f"""
{problem.strip()}

请直接解答这个问题。
"""
    
    messages = [{"role": "user", "content": zero_shot_prompt}]
    
    try:
        result = pipe(messages, max_new_tokens=300, temperature=0.2, do_sample=True)
        print("零样本提示结果:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"零样本提示执行出错: {e}")
    
    # --- 4. 少样本提示 ---
    print("\n" + "="*60)
    print("方法2: 少样本提示 (Few-shot)")
    print("="*60)
    
    few_shot_prompt = f"""
学习以下数学问题的解答方法：

问题1：一个班级有40名学生，其中50%是男生。如果班级中30%的学生参加了体育比赛，且参加比赛的学生中男女比例是3:2，那么有多少名男生参加了体育比赛？
解答：
- 班级总人数：40人
- 男生人数：40 × 50% = 20人
- 参加比赛的学生：40 × 30% = 12人
- 参加比赛的男生：12 × 3/(3+2) = 12 × 3/5 = 7.2 ≈ 7人
答案：7名男生参加了体育比赛

问题2：一个班级有50名学生，其中40%是女生。如果班级中60%的学生参加了科学竞赛，且参加竞赛的学生中男女比例是1:2，那么有多少名女生参加了科学竞赛？
解答：
- 班级总人数：50人
- 女生人数：50 × 40% = 20人
- 参加竞赛的学生：50 × 60% = 30人
- 参加竞赛的女生：30 × 1/(1+2) = 30 × 1/3 = 10人
答案：10名女生参加了科学竞赛

现在请用相同的方法解答：
{problem.strip()}
"""
    
    messages = [{"role": "user", "content": few_shot_prompt}]
    
    try:
        result = pipe(messages, max_new_tokens=400, temperature=0.2, do_sample=True)
        print("少样本提示结果:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"少样本提示执行出错: {e}")
    
    # --- 5. 思维链提示 ---
    print("\n" + "="*60)
    print("方法3: 思维链提示 (Chain-of-Thought)")
    print("="*60)
    
    cot_prompt = f"""
{problem.strip()}

请逐步思考并解答这个问题，展示你的推理过程：
1. 首先分析题目给出的信息
2. 然后计算每一步的结果
3. 最后得出最终答案
"""
    
    messages = [{"role": "user", "content": cot_prompt}]
    
    try:
        result = pipe(messages, max_new_tokens=500, temperature=0.2, do_sample=True)
        print("思维链提示结果:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"思维链提示执行出错: {e}")
    
    # --- 6. 角色扮演提示 ---
    print("\n" + "="*60)
    print("方法4: 角色扮演提示 (Role-playing)")
    print("="*60)
    
    role_play_prompt = f"""
你是一位经验丰富的数学老师，擅长用简单易懂的方法教授数学问题。

{problem.strip()}

请以数学老师的身份，用清晰的教学方式解答这个问题，包括：
1. 分析题目关键信息
2. 展示解题思路
3. 逐步计算过程
4. 验证答案的正确性
"""
    
    messages = [{"role": "user", "content": role_play_prompt}]
    
    try:
        result = pipe(messages, max_new_tokens=500, temperature=0.2, do_sample=True)
        print("角色扮演提示结果:")
        print(result[0]['generated_text'])
    except Exception as e:
        print(f"角色扮演提示执行出错: {e}")
    
    # --- 7. 方法对比分析 ---
    print("\n" + "="*60)
    print("方法对比分析")
    print("="*60)
    
    print("""
提示工程方法对比分析：

1. 零样本提示 (Zero-shot):
   - 优点：简单直接，无需准备示例
   - 缺点：对复杂问题可能理解不准确
   - 适用场景：简单、通用的任务

2. 少样本提示 (Few-shot):
   - 优点：通过示例引导，提高准确性
   - 缺点：需要准备高质量的示例
   - 适用场景：有明确格式要求的任务

3. 思维链提示 (Chain-of-Thought):
   - 优点：强制模型逐步推理，提高逻辑性
   - 缺点：可能产生冗余的推理过程
   - 适用场景：需要逻辑推理的复杂问题

4. 角色扮演提示 (Role-playing):
   - 优点：激活模型的专业知识，回答更专业
   - 缺点：可能过于冗长，不够简洁
   - 适用场景：需要专业知识的领域问题
""")

def main():
    """主函数：运行提示工程演示"""
    
    print("提示工程实战 - 参考答案")
    print("="*60)
    
    try:
        prompt_engineering_demo()
    except Exception as e:
        print(f"提示工程演示执行出错: {e}")
    
    print("\n" + "="*60)
    print("提示工程演示完成！")
    print("="*60)

if __name__ == "__main__":
    main()
