"""
对话系统使用示例
演示如何直接使用各个模块
"""

from dialogue_system import TaskOrientedDialogue, LLMChat, NLU, DM, NLG


def example_task_oriented():
    """任务型对话系统示例"""
    print("=" * 60)
    print("任务型对话系统示例")
    print("=" * 60)
    
    dialogue = TaskOrientedDialogue()
    
    # 示例对话
    examples = [
        "你好",
        "我想查询北京的天气",
        "我想预订餐厅",
        "明天晚上7点",
        "3个人",
        "现在几点了？",
        "再见"
    ]
    
    for user_input in examples:
        print(f"\n用户: {user_input}")
        response = dialogue.process(user_input)
        print(f"助手: {response}")


def example_llm_chat():
    """LLM闲聊系统示例"""
    print("\n" + "=" * 60)
    print("LLM闲聊系统示例（模拟模式）")
    print("=" * 60)
    
    chat = LLMChat()  # 不提供API密钥，使用模拟模式
    
    examples = [
        "你好，今天心情不错",
        "你觉得人工智能的未来会怎样？",
        "给我讲个笑话吧",
        "谢谢你的帮助"
    ]
    
    for user_input in examples:
        print(f"\n用户: {user_input}")
        response = chat.chat(user_input)
        print(f"助手: {response}")


def example_module_detailed():
    """详细展示各个模块的工作过程"""
    print("\n" + "=" * 60)
    print("模块详细工作过程示例")
    print("=" * 60)
    
    user_input = "我想查询北京的天气"
    print(f"\n用户输入: {user_input}")
    
    # NLU模块
    nlu = NLU()
    nlu_result = nlu.parse(user_input)
    print(f"\n[NLU结果]")
    print(f"  意图: {nlu_result['intent']}")
    print(f"  实体: {nlu_result['entities']}")
    print(f"  置信度: {nlu_result['confidence']}")
    
    # DM模块
    dm = DM()
    dm_result = dm.update_state(nlu_result)
    print(f"\n[DM结果]")
    print(f"  动作: {dm_result['action']}")
    print(f"  状态: {dm_result['state']}")
    
    # NLG模块
    nlg = NLG()
    response = nlg.generate(dm_result['action'], dm_result['state'])
    print(f"\n[NLG结果]")
    print(f"  回复: {response}")


if __name__ == "__main__":
    # 运行所有示例
    example_task_oriented()
    example_llm_chat()
    example_module_detailed()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)

