# 多模态数据分析师Agent

一个基于AI的多模态数据分析系统，能够理解图表图片、分析CSV数据并生成专业的数据分析报告。

## 功能特性

- 📊 **图表识别**：使用多模态AI模型理解各种类型的图表和数据可视化
- 📄 **CSV分析**：自动分析CSV文件，提取统计信息和数据洞察
- 📋 **报告生成**：结合图表和CSV数据，生成专业的数据分析报告
- 🌐 **Web界面**：简洁美观的Web界面，支持拖拽上传文件

## 技术栈

- **FastAPI**: 后端API框架
- **LangChain**: LLM应用框架
- **OpenAI/通义千问**: 多模态AI模型
- **Pandas**: 数据处理
- **HTML/CSS/JavaScript**: Web前端

## 安装步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件：

```env
# LLM配置
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.openai.com/v1  # 或通义千问的API地址
LLM_MODEL=gpt-4-vision-preview  # 或 qwen-vl-max (通义千问多模态模型)

# API服务配置
API_HOST=127.0.0.1
API_PORT=8000
```

### 3. 启动服务

```bash
python api_server.py
```

### 4. 访问Web界面

在浏览器中打开：`http://127.0.0.1:8000`

## 停止服务器

### 方法1：使用停止脚本（推荐）

**Windows:**
```bash
stop_server.bat
```

**跨平台（Python）:**
```bash
python stop_server.py
```

**指定端口:**
```bash
python stop_server.py --port 8001
```

### 方法2：在运行终端中按 Ctrl+C

如果服务器是在终端前台运行的，直接按 `Ctrl+C` 即可停止。

### 方法3：手动查找并关闭进程

**Windows:**
```bash
# 查找占用端口的进程
netstat -ano | findstr :8000

# 关闭进程（替换PID为实际进程ID）
taskkill /PID [进程ID] /F
```

**Linux/Mac:**
```bash
# 查找占用端口的进程
lsof -ti :8000

# 关闭进程
kill -9 [进程ID]
```

## 使用方法

1. **上传图表图片**：点击左侧上传区域，选择图表图片（支持常见图片格式）
2. **上传CSV文件**：点击右侧上传区域，选择CSV数据文件
3. **开始分析**：点击"开始分析"按钮
4. **查看报告**：等待分析完成后，查看生成的数据分析报告

## API接口

### POST /analyze

分析数据并生成报告

**请求参数：**
- `image` (file, optional): 图表图片文件
- `csv` (file, optional): CSV数据文件

**响应示例：**
```json
{
  "report": "数据分析报告内容...",
  "chart_analysis": "图表分析结果...",
  "csv_analysis": "CSV数据分析结果..."
}
```

### GET /health

健康检查接口

## 项目结构

```
Multimodal_Agent/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Agent基类
│   └── data_analyst_agent.py  # 数据分析师Agent
├── core/
│   ├── __init__.py
│   ├── controller.py          # MCP控制器
│   └── message_bus.py         # 消息总线
├── api_server.py              # API服务器
├── stop_server.py             # 停止服务器脚本（Python）
├── stop_server.bat            # 停止服务器脚本（Windows批处理）
├── test_api.py                # API测试脚本
├── create_sample_data.py      # 创建示例数据
├── requirements.txt           # 依赖包
├── README.md                  # 项目说明
└── .env                       # 环境变量配置
```

## 支持的模型

- **OpenAI**: GPT-4 Vision, GPT-4 Turbo
- **通义千问**: qwen-vl-max, qwen-vl-plus

## 注意事项

1. 确保API密钥配置正确
2. 图片文件大小建议不超过10MB
3. CSV文件编码建议使用UTF-8
4. 首次运行可能需要下载模型，请耐心等待

## 许可证

MIT License
