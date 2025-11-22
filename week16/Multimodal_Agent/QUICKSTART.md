# 快速开始指南

## 1. 环境准备

### 安装Python依赖

```bash
cd Multimodal_Agent
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并配置：

```bash
# Windows
copy .env.example .env

# Linux/Mac
cp .env.example .env
```

编辑 `.env` 文件，填入你的API密钥：

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4-vision-preview
```

## 2. 启动服务

```bash
python api_server.py
```

看到以下输出表示启动成功：

```
============================================================
  多模态数据分析师Agent API服务器
============================================================
  服务地址: http://127.0.0.1:8000
  Web界面: http://127.0.0.1:8000/
============================================================
```

## 3. 使用Web界面

1. 在浏览器中打开 `http://127.0.0.1:8000`
2. 上传图表图片（可选）
3. 上传CSV文件（可选）
4. 点击"开始分析"按钮
5. 等待分析完成，查看报告

## 4. 使用API接口

### 使用curl测试

```bash
# 分析图片和CSV
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "image=@chart.png" \
  -F "csv=@data.csv"
```

### 使用Python

```python
import requests

# 上传文件并分析
files = {
    'image': open('chart.png', 'rb'),
    'csv': open('data.csv', 'rb')
}

response = requests.post('http://127.0.0.1:8000/analyze', files=files)
result = response.json()

print(result['report'])
```

## 5. 支持的模型

### OpenAI
- `gpt-4-vision-preview` - 推荐用于图表分析
- `gpt-4-turbo` - 通用多模态模型

### 通义千问
- `qwen-vl-max` - 通义千问多模态模型
- `qwen-vl-plus` - 通义千问多模态模型（更快）

配置示例：
```env
QWEN_API_KEY=your_qwen_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-vl-max
```

## 6. 功能演示

### 示例1：仅分析图表

```python
from agents.data_analyst_agent import DataAnalystAgent
from core.controller import Controller

controller = Controller()
agent = DataAnalystAgent(
    name="Analyst",
    controller_reference=controller
)

# 分析图表
result = agent.analyze_chart(image_path="chart.png")
print(result['analysis'])
```

### 示例2：仅分析CSV

```python
result = agent.analyze_csv(csv_path="data.csv")
print(result['analysis'])
```

### 示例3：综合分析

```python
result = agent.generate_report(
    image_path="chart.png",
    csv_path="data.csv"
)
print(result['report'])
```

## 7. 常见问题

### Q: 图片上传失败？
A: 检查图片格式和大小，建议使用JPG、PNG格式，大小不超过10MB

### Q: CSV分析出错？
A: 确保CSV文件使用UTF-8编码，包含表头行

### Q: API调用超时？
A: 增加超时时间，或使用更快的模型（如qwen-vl-plus）

### Q: 模型不支持多模态？
A: 确保使用支持视觉的模型（gpt-4-vision或qwen-vl系列）

## 8. 下一步

- 查看 `README.md` 了解详细功能
- 查看 `agents/data_analyst_agent.py` 了解Agent实现
- 自定义报告模板和提示词

