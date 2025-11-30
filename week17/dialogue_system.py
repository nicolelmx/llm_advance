"""
对话系统主程序
支持任务型对话和LLM API闲聊两种模式
"""

from typing import Dict, List, Optional
import json
import re


class NLU:
    """自然语言理解模块 (Natural Language Understanding)"""
    
    def __init__(self):
        # 意图识别规则 - 扩展更多表达方式
        self.intent_patterns = {
            'greeting': [
                r'你好', r'hello', r'hi', r'早上好', r'下午好', r'晚上好',
                r'问候', r'您好', r'嗨', r'hey', r'greeting'
            ],
            'weather_query': [
                r'天气', r'温度', r'下雨', r'晴天', r'weather', r'气候',
                r'查询天气', r'天气怎么样', r'天气如何', r'天气预报'
            ],
            'restaurant_booking': [
                r'订餐', r'餐厅', r'预订', r'吃饭', r'book', r'预约',
                r'订座', r'订位', r'订饭店', r'订餐厅', r'预约餐厅'
            ],
            'time_query': [
                r'时间', r'几点', r'现在', r'time', r'查询时间',
                r'现在几点', r'什么时候', r'当前时间'
            ],
            'goodbye': [
                r'再见', r'拜拜', r'bye', r'goodbye', r'再会',
                r'拜', r'88', r'退出', r'结束'
            ]
        }
        
        # 中国主要城市列表（扩展）
        self.cities = [
            '北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '重庆',
            '武汉', '西安', '天津', '苏州', '长沙', '郑州', '青岛', '大连',
            '宁波', '厦门', '福州', '济南', '合肥', '石家庄', '哈尔滨', '长春',
            '沈阳', '昆明', '贵阳', '南宁', '海口', '太原', '南昌', '乌鲁木齐',
            '拉萨', '银川', '西宁', '呼和浩特', '兰州', 'beijing', 'shanghai',
            'hangzhou', 'guangzhou', 'shenzhen'
        ]
        
        # 省份和城市组合模式
        self.province_city_pattern = r'([^省]+省)?([^市]+市)'
        
        # 实体提取规则
        self.entity_patterns = {
            'location': self.cities + [self.province_city_pattern],
            'time': [
                r'\d+点', r'\d+:\d+', r'今天', r'明天', r'后天', r'大后天',
                r'上午', r'下午', r'晚上', r'中午', r'傍晚',
                r'今晚', r'明晚', r'今晚\d+点', r'明天\d+点'
            ],
            'number': [
                r'\d+人', r'\d+位', r'\d+个', r'\d+名', r'一个人', r'两个人',
                r'三人', r'四人', r'五人', r'六人', r'七人', r'八人', r'九人', r'十人'
            ]
        }
    
    def parse(self, user_input: str, context: List[Dict] = None) -> Dict:
        """解析用户输入，返回意图和实体"""
        user_input_lower = user_input.lower()
        user_input_original = user_input
        
        # 意图识别
        intent = 'unknown'
        confidence = 0.0
        
        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    intent = intent_name
                    confidence = 0.8
                    break
            if intent != 'unknown':
                break
        
        # 如果意图未知，检查上下文
        if intent == 'unknown' and context:
            # 如果上下文中有未完成的天气查询，当前输入可能是城市名
            last_intent = None
            for ctx in reversed(context[-3:]):  # 检查最近3轮对话
                if ctx.get('intent') == 'weather_query':
                    last_intent = 'weather_query'
                    break
            
            if last_intent == 'weather_query':
                # 尝试将当前输入识别为城市
                location = self._extract_location(user_input_original)
                if location:
                    intent = 'weather_query'
                    confidence = 0.7
        
        # 实体提取
        entities = {}
        
        # 提取位置
        location = self._extract_location(user_input_original)
        if location:
            entities['location'] = location
        
        # 提取时间
        for pattern in self.entity_patterns['time']:
            match = re.search(pattern, user_input_original)
            if match:
                entities['time'] = match.group()
                break
        
        # 提取人数
        for pattern in self.entity_patterns['number']:
            match = re.search(pattern, user_input_original)
            if match:
                entities['number'] = match.group()
                break
        
        return {
            'intent': intent,
            'entities': entities,
            'confidence': confidence,
            'original_text': user_input
        }
    
    def _extract_location(self, text: str) -> Optional[str]:
        """提取城市名称"""
        # 先尝试匹配省份+城市模式
        match = re.search(self.province_city_pattern, text)
        if match:
            city = match.group(2) if match.group(2) else match.group(0)
            # 提取城市名（去掉"市"字）
            city = city.replace('市', '').strip()
            return city
        
        # 直接匹配城市列表
        for city in self.cities:
            if city in text:
                return city.replace('市', '').strip()
        
        # 尝试提取常见城市名（2-3个字符，以"市"结尾）
        city_match = re.search(r'([\u4e00-\u9fa5]{2,3})市', text)
        if city_match:
            return city_match.group(1)
        
        return None


class DM:
    """对话管理模块 (Dialogue Management)"""
    
    def __init__(self):
        self.dialogue_state = {
            'intent': None,
            'slots': {},
            'context': []
        }
    
    def update_state(self, nlu_result: Dict) -> Dict:
        """更新对话状态并决定下一步动作"""
        intent = nlu_result['intent']
        entities = nlu_result['entities']
        
        # 如果意图未知，尝试从上下文推断
        if intent == 'unknown' and self.dialogue_state['context']:
            # 检查是否有未完成的任务
            last_state = self.dialogue_state['context'][-1] if self.dialogue_state['context'] else {}
            last_intent = last_state.get('intent')
            
            # 如果上一轮是天气查询但没有城市，当前输入可能是城市
            if last_intent == 'weather_query' and 'location' not in self.dialogue_state['slots']:
                if 'location' in entities:
                    intent = 'weather_query'
                    nlu_result['intent'] = intent
                    nlu_result['confidence'] = 0.7
                else:
                    # 即使没有提取到location实体，如果上下文需要，也尝试将整个输入作为城市
                    user_input = nlu_result.get('original_text', '').strip()
                    if user_input and len(user_input) <= 10:  # 可能是城市名
                        # 检查是否可能是城市名（2-4个中文字符）
                        if re.match(r'^[\u4e00-\u9fa5]{2,4}$', user_input):
                            entities['location'] = user_input
                            intent = 'weather_query'
                            nlu_result['intent'] = intent
                            nlu_result['entities'] = entities
                            nlu_result['confidence'] = 0.6
            
            # 如果上一轮是餐厅预订，当前输入可能是时间或人数
            elif last_intent == 'restaurant_booking':
                if 'time' in entities or 'number' in entities:
                    intent = 'restaurant_booking'
                    nlu_result['intent'] = intent
                    nlu_result['confidence'] = 0.7
        
        # 更新意图
        if intent != 'unknown':
            self.dialogue_state['intent'] = intent
        
        # 更新槽位
        self.dialogue_state['slots'].update(entities)
        
        # 保存上下文
        self.dialogue_state['context'].append(nlu_result)
        
        # 决定动作
        action = self._decide_action(intent, entities)
        
        return {
            'action': action,
            'state': self.dialogue_state.copy()
        }
    
    def _decide_action(self, intent: str, entities: Dict) -> str:
        """根据意图和实体决定动作"""
        if intent == 'greeting':
            return 'greet'
        elif intent == 'weather_query':
            if 'location' in entities:
                return 'query_weather'
            else:
                return 'request_location'
        elif intent == 'restaurant_booking':
            required_slots = ['time', 'number']
            missing_slots = [slot for slot in required_slots if slot not in self.dialogue_state['slots']]
            if missing_slots:
                return f'request_{missing_slots[0]}'
            else:
                return 'confirm_booking'
        elif intent == 'time_query':
            return 'tell_time'
        elif intent == 'goodbye':
            return 'say_goodbye'
        else:
            return 'clarify'
    
    def reset(self):
        """重置对话状态"""
        self.dialogue_state = {
            'intent': None,
            'slots': {},
            'context': []
        }


class NLG:
    """自然语言生成模块 (Natural Language Generation)"""
    
    def __init__(self):
        self.templates = {
            'greet': [
                "你好！我是智能助手，可以帮你查询天气、预订餐厅等。有什么可以帮你的吗？",
                "你好！很高兴为你服务。",
                "你好！我能为你做什么？"
            ],
            'query_weather': lambda slots: f"根据查询，{slots.get('location', '当地')}今天天气晴朗，温度25度。",
            'request_location': "请问你想查询哪个城市的天气？",
            'request_time': "请问你想预订什么时间？",
            'request_number': "请问有几个人用餐？",
            'confirm_booking': lambda slots: f"好的，已为你预订{slots.get('time', '指定时间')}，{slots.get('number', '指定人数')}人的餐厅。",
            'tell_time': "现在是下午3点。",
            'say_goodbye': [
                "再见！祝你愉快！",
                "再见！期待下次为你服务。",
                "拜拜！"
            ],
            'clarify': "抱歉，我没有理解你的意思。你能再说一遍吗？"
        }
    
    def generate(self, action: str, state: Dict = None) -> str:
        """根据动作生成回复"""
        if action not in self.templates:
            return "抱歉，我无法处理这个请求。"
        
        template = self.templates[action]
        
        if isinstance(template, list):
            import random
            return random.choice(template)
        elif callable(template):
            return template(state.get('slots', {}))
        else:
            return template


class TaskOrientedDialogue:
    """任务型对话系统"""
    
    def __init__(self):
        self.nlu = NLU()
        self.dm = DM()
        self.nlg = NLG()
    
    def process(self, user_input: str) -> str:
        """处理用户输入并返回回复"""
        # NLU: 理解用户输入（传入上下文）
        context = self.dm.dialogue_state.get('context', [])
        nlu_result = self.nlu.parse(user_input, context)
        
        # DM: 管理对话状态
        dm_result = self.dm.update_state(nlu_result)
        
        # NLG: 生成回复
        response = self.nlg.generate(dm_result['action'], dm_result['state'])
        
        return response
    
    def reset(self):
        """重置对话"""
        self.dm.reset()


class LLMChat:
    """基于LLM的闲聊系统"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-turbo"):
        """
        初始化LLM聊天系统
        
        Args:
            api_key: QWEN API密钥（如果使用QWEN）
            model: 使用的模型名称，默认为qwen-turbo
        """
        self.api_key = api_key
        self.model = model
        self.conversation_history = []
        self.use_qwen = api_key is not None
    
    def chat(self, user_input: str) -> str:
        """与LLM进行对话"""
        if self.use_qwen:
            return self._chat_with_qwen(user_input)
        else:
            return self._chat_with_mock_llm(user_input)
    
    def _chat_with_qwen(self, user_input: str) -> str:
        """使用QWEN API进行对话"""
        try:
            import dashscope
            from dashscope import Generation
            
            dashscope.api_key = self.api_key
            
            # 添加用户消息到历史
            self.conversation_history.append({
                "role": "user",
                "content": user_input
            })
            
            # 构建消息列表
            messages = [
                {"role": "system", "content": "你是一个友好、 helpful的AI助手。"}
            ] + self.conversation_history[-10:]  # 只保留最近10轮对话
            
            # 调用QWEN API
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                result_format='message'
            )
            
            if response.status_code == 200:
                assistant_reply = response.output.choices[0].message.content
                
                # 添加助手回复到历史
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_reply
                })
                
                return assistant_reply
            else:
                return f"抱歉，调用QWEN API时出错: {response.message}"
            
        except ImportError:
            print("警告: 未安装dashscope库，使用模拟LLM")
            print("请运行: pip install dashscope")
            return self._chat_with_mock_llm(user_input)
        except Exception as e:
            return f"抱歉，调用QWEN API时出错: {str(e)}"
    
    def _chat_with_mock_llm(self, user_input: str) -> str:
        """模拟LLM回复（用于演示，不需要API密钥）"""
        # 简单的规则回复，模拟LLM行为
        responses = {
            'greeting': "你好！很高兴和你聊天！",
            'question': "这是一个有趣的问题。让我想想...",
            'emotion': "我理解你的感受。",
            'default': "我明白了。能告诉我更多细节吗？"
        }
        
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['你好', 'hello', 'hi']):
            return responses['greeting']
        elif '?' in user_input or '？' in user_input:
            return responses['question']
        elif any(word in user_lower for word in ['开心', '难过', '生气', '高兴']):
            return responses['emotion']
        else:
            # 生成一个基于输入的简单回复
            return f"关于「{user_input}」，我觉得这很有意思。你能详细说说吗？"
    
    def reset(self):
        """重置对话历史"""
        self.conversation_history = []


def main():
    """主程序入口"""
    print("=" * 60)
    print("对话系统演示程序")
    print("=" * 60)
    print("\n请选择模式:")
    print("1. 任务型对话系统（规则-based）")
    print("2. LLM API 闲聊系统")
    print("3. 退出")
    
    choice = input("\n请输入选项 (1/2/3): ").strip()
    
    if choice == '1':
        run_task_oriented_dialogue()
    elif choice == '2':
        run_llm_chat()
    elif choice == '3':
        print("再见！")
        return
    else:
        print("无效选项，请重新运行程序。")
        return


def run_task_oriented_dialogue():
    """运行任务型对话系统"""
    print("\n" + "=" * 60)
    print("任务型对话系统")
    print("=" * 60)
    print("支持功能: 问候、查询天气、预订餐厅、查询时间")
    print("输入 'quit' 退出\n")
    
    dialogue = TaskOrientedDialogue()
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', '退出', 'exit']:
            print("对话已结束。")
            break
        
        if not user_input:
            continue
        
        response = dialogue.process(user_input)
        print(f"助手: {response}\n")


def run_llm_chat():
    """运行LLM闲聊系统"""
    print("\n" + "=" * 60)
    print("LLM API 闲聊系统 (QWEN)")
    print("=" * 60)
    
    # 询问是否使用QWEN API
    use_api = input("是否使用QWEN API? (y/n，默认n): ").strip().lower()
    
    api_key = None
    model = "qwen-turbo"
    if use_api == 'y':
        api_key = input("请输入QWEN API密钥 (DashScope API Key): ").strip()
        if not api_key:
            print("未提供API密钥，将使用模拟LLM模式。")
        else:
            model_choice = input("选择模型 (1: qwen-turbo, 2: qwen-plus, 3: qwen-max，默认1): ").strip()
            if model_choice == '2':
                model = "qwen-plus"
            elif model_choice == '3':
                model = "qwen-max"
    
    chat = LLMChat(api_key=api_key, model=model)
    
    print("\n开始聊天（输入 'quit' 退出）\n")
    
    while True:
        user_input = input("你: ").strip()
        
        if user_input.lower() in ['quit', '退出', 'exit']:
            print("对话已结束。")
            break
        
        if not user_input:
            continue
        
        print("助手: ", end="", flush=True)
        response = chat.chat(user_input)
        print(f"{response}\n")


if __name__ == "__main__":
    main()

