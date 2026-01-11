# Project_B - 智能风控预测与解释系统

## 项目简介

Project_B 是一个整合了传统机器学习模型（LightGBM）和本地大语言模型（Ollama qwen3:4b）的智能风控系统。

### 核心功能
1. **欺诈预测**：使用LightGBM模型预测信用卡交易是否为欺诈
2. **SHAP解释**：分析哪些特征对预测结果影响最大
3. **LLM解释**：使用qwen3:4b生成通俗易懂的中文解释报告
4. **策略建议**：针对高风险交易生成风控策略建议

### 技术栈
- **预测模型**：LightGBM（来自Project_A）
- **解释分析**：SHAP
- **LLM服务**：Ollama + qwen3:4b
- **API框架**：FastAPI
- **部署方式**：本地部署

## 快速开始

### 1. 前置条件
- ✅ 已安装Ollama并拉取qwen3:4b模型
- ✅ Project_A已训练好模型（`models/lgbm_model.pkl`）

### 2. 安装依赖
```bash
cd EDU/Project_B
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 3. 启动服务
```bash
# 方式1：使用启动脚本
python start_api.py

# 方式2：直接运行
python src/api/app.py
```

### 4. 访问API
- API文档：http://localhost:8001/docs
- 健康检查：http://localhost:8001/health

## 详细文档

请查看 [指导手册.md](./指导手册.md) 获取完整的使用说明。

## 二、前置条件
1. 已安装 Ollama，并已拉取模型：
   ```bash
   ollama list   # 应看到 qwen3:4b
   ```
2. 确保 Ollama 服务在本机运行（通常安装后自启动），默认 API 地址：
   ```
   http://localhost:11434
   ```

## 三、快速开始
```bash
cd EDU/Project_B
python ollama_chat.py --prompt "你好，介绍一下这个项目"
```
可选参数：
- `--model`  默认 `qwen3:4b`，可换 `qwen3-vl:4b`（多模态）或其他已拉取模型。
- `--system` 系统提示词（设定模型角色/风格）。
- `--stream` 开启流式输出。

示例：
```bash
# 基本对话
python ollama_chat.py --prompt "帮我总结一下信用卡欺诈预测项目的目标"

# 指定系统提示
python ollama_chat.py --prompt "列出三点可落地的改进建议" \
  --system "你是资深风控算法工程师，请给出可执行的落地建议"

# 流式输出
python ollama_chat.py --prompt "用 3 条 bullet 总结本项目" --stream
```

## 四、脚本说明（ollama_chat.py）
- 调用地址：`http://localhost:11434/api/chat`
- 请求体：
  ```json
  {
    "model": "qwen3:4b",
    "messages": [
      {"role": "system", "content": "..."},
      {"role": "user", "content": "..."}
    ],
    "stream": false
  }
  ```
- 响应：若 `stream=false`，一次性返回；若 `stream=true`，逐行增量返回内容。

## 五、常见问题
1) 模型未找到 / 报 404  
   - 运行 `ollama list` 确认模型存在；不存在则 `ollama pull qwen3:4b`。

2) 无法连接 11434 端口  
   - 确认 Ollama 服务已启动；若端口被占用，可在 `ollama.yaml` 中调整，或用 `--addr` 启动。

3) 想用多模态（图片）  
   - 使用 `qwen3-vl:4b` 等多模态模型，API 需传递 `images`（Base64）；本脚本未内置，可按 Ollama API 文档扩展。

4) 想做 RAG / Agent  
   - 先用本脚本验证模型可用；后续可在上层应用中集成检索、工具调用，再把拼好的 prompt 发给 `/api/chat`。

## 六、目录
- `ollama_chat.py`：调用本地 Ollama 模型的对话脚本
- （可扩展）增加 RAG/Agent 逻辑或前端界面时，可在本目录新增相应文件

## 七、下一步可扩展
- 支持多模态输入（图文混合），调用 `qwen3-vl:4b`。
- 增加 RAG 示例：本地向量库 + prompt 构造 + `/api/chat`。
- 增加批量推理/评测脚本，统计响应质量与延迟。

