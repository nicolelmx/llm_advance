"""
使用本地 Ollama 的 qwen3:4b 模型进行对话

先确保本地已安装并拉取模型：
    ollama list          # 确认 qwen3:4b 已存在
    # 如未启动，运行 ollama serve（通常已随 Ollama 自启）

运行示例：
    python ollama_chat.py --prompt "你好，介绍一下这个项目"

可选参数：
    --model     模型名称（默认 qwen3:4b）
    --system    系统提示词
    --stream    开启流式输出
"""

import argparse
import json
import sys
import requests
from requests.exceptions import ConnectionError, Timeout, RequestException


def chat(model: str, prompt: str, system: str = "", stream: bool = False):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system} if system else None,
            {"role": "user", "content": prompt},
        ],
        "stream": stream,
    }
    # 过滤掉 None 的 system
    payload["messages"] = [m for m in payload["messages"] if m]

    try:
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
    except ConnectionError as e:
        print("❌ 错误：无法连接到 Ollama 服务", file=sys.stderr)
        print(f"\n可能的原因：", file=sys.stderr)
        print("1. Ollama 服务未启动", file=sys.stderr)
        print("2. Ollama 服务未在默认端口 11434 上运行", file=sys.stderr)
        print("3. 防火墙阻止了连接", file=sys.stderr)
        print(f"\n解决方法：", file=sys.stderr)
        print("1. 检查 Ollama 是否运行：", file=sys.stderr)
        print("   ollama list", file=sys.stderr)
        print("2. 如果未运行，启动 Ollama 服务：", file=sys.stderr)
        print("   ollama serve", file=sys.stderr)
        print("3. 测试连接：", file=sys.stderr)
        print("   curl http://localhost:11434/api/tags", file=sys.stderr)
        print(f"\n详细错误信息：{e}", file=sys.stderr)
        sys.exit(1)
    except Timeout as e:
        print("❌ 错误：请求超时（超过 300 秒）", file=sys.stderr)
        print("模型响应时间过长，请检查模型是否正常工作", file=sys.stderr)
        print(f"\n详细错误信息：{e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        # HTTPError 异常对象包含 response 属性
        if e.response.status_code == 404:
            print(f"❌ 错误：模型 '{model}' 未找到", file=sys.stderr)
            print(f"\n解决方法：", file=sys.stderr)
            print(f"1. 检查模型是否存在：ollama list", file=sys.stderr)
            print(f"2. 如果不存在，拉取模型：ollama pull {model}", file=sys.stderr)
        else:
            print(f"❌ HTTP 错误：{e.response.status_code}", file=sys.stderr)
            print(f"响应内容：{e.response.text}", file=sys.stderr)
        sys.exit(1)
    except RequestException as e:
        print(f"❌ 请求错误：{e}", file=sys.stderr)
        sys.exit(1)

    if stream:
        # 流式响应：逐行打印 content
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line.decode("utf-8"))
            delta = data.get("message", {}).get("content", "")
            if delta:
                sys.stdout.write(delta)
                sys.stdout.flush()
        print()
    else:
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        print(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True, help="用户输入")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama 模型名称")
    parser.add_argument("--system", default="", help="系统提示词")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    args = parser.parse_args()

    chat(model=args.model, prompt=args.prompt, system=args.system, stream=args.stream)


if __name__ == "__main__":
    main()

