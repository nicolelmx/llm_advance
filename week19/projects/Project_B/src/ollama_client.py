"""
Ollama API 客户端封装
"""
import json
import requests
from typing import Optional, List, Dict

from plotly.matplotlylib.mplexporter.utils import image_to_base64


class OllamaClient:
    """Ollama API 客户端"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """
        初始化客户端
        
        Args:
            base_url: Ollama服务地址
        """
        self.base_url = base_url.rstrip("/")
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
    
    def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        system: Optional[str] = None,
        image_path: Optional[str] = None,
        timeout: int = 60  # 减少超时时间到60秒
    ) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型名称
            messages: 消息列表，格式：[{"role": "user", "content": "..."}]
            stream: 是否流式输出
            system: 系统提示词（可选）
            timeout: 超时时间（秒）
        
        Returns:
            str: 模型回复内容
        """
        payload = {
            "model": model,
            "messages": messages.copy(),
            "stream": stream
        }
        
        if system:
            payload["messages"].insert(0, {"role": "system", "content": system})

        # 1. 图片转Base64
        if image_path:
            image_base64 = image_to_base64(image_path)
        
        resp = requests.post(self.chat_url, json=payload, timeout=timeout)
        resp.raise_for_status()
        
        if stream:
            content = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                delta = data.get("message", {}).get("content", "")
                if delta:
                    content += delta
            return content
        else:
            data = resp.json()
            return data.get("message", {}).get("content", "")
    
    def explain_prediction(
        self,
        model: str,
        explanation_text: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        使用LLM解释预测结果
        
        Args:
            model: 模型名称
            explanation_text: SHAP解释文本
            system_prompt: 系统提示词（可选）
        
        Returns:
            str: LLM生成的解释报告
        """
        if system_prompt is None:
            system_prompt = """你是一个专业的风险控制分析师。请根据提供的模型预测结果和特征重要性分析，生成一份清晰、专业的中文解释报告。

要求：
1. 用通俗易懂的语言解释为什么模型给出这个预测结果
2. 重点说明哪些特征对预测结果影响最大
3. 给出风险提示或建议
4. 报告要简洁明了，控制在200字以内"""
        
        user_prompt = f"""请分析以下模型预测结果，并生成一份解释报告：

{explanation_text}

请用中文生成一份专业的解释报告。"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        return self.chat(model=model, messages=messages, system=system_prompt, stream=False)
    
    def generate_strategy(
        self,
        model: str,
        risk_summary: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        生成风控策略建议
        
        Args:
            model: 模型名称
            risk_summary: 风险摘要
            system_prompt: 系统提示词（可选）
        
        Returns:
            str: 策略建议
        """
        if system_prompt is None:
            system_prompt = """你是一个资深的风控策略专家。请根据提供的风险分析结果，给出可执行的运营和风控策略建议。

要求：
1. 建议要具体、可执行
2. 考虑实际业务场景
3. 给出3-5条建议
4. 用中文输出"""
        
        user_prompt = f"""根据以下风险分析结果，请给出风控策略建议：

{risk_summary}

请给出3-5条可执行的策略建议。"""
        
        messages = [{"role": "user", "content": user_prompt}]
        
        return self.chat(model=model, messages=messages, system=system_prompt, stream=False)
    
    def health_check(self) -> bool:
        """
        检查Ollama服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except:
            return False


if __name__ == "__main__":
    # 测试
    client = OllamaClient()
    
    # 健康检查
    if client.health_check():
        print("✓ Ollama服务可用")
    else:
        print("✗ Ollama服务不可用，请检查服务是否启动")
        exit(1)
    
    # 测试对话
    response = client.chat(
        model="qwen3:4b",
        messages=[{"role": "user", "content": "你好，请用一句话介绍你自己"}]
    )
    print("\n模型回复：", response)

