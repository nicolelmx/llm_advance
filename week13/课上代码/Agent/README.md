## 五子棋 AI 自主下棋 Agent 项目

### 功能概述

基于 **LangChain ReAct 架构**构建的五子棋 AI 智能体，能够：
- 🎮 自主思考并下棋（AI vs AI 完整对局）
- 🤖 与人类对局（AI 作为对手）
- 📊 在线下载棋谱数据集并学习开局模式
- 🧠 智能评估局面并推荐最佳走法
- 💾 保存和加载游戏状态

### 技术架构

- **框架**: LangChain ReAct（Reasoning + Acting）
- **LLM**: Qwen（兼容 OpenAI 接口）
- **游戏**: 五子棋（15x15 标准棋盘）
- **工具集**: 游戏管理、局面评估、数据集下载等

### 目录结构

```
Agent/
  README.md
  requirements.txt
  .env.example
  __init__.py
  config.py                    # 配置加载
  agent_builder.py            # Agent 构建
  run_demo.py                 # 演示脚本
  tools/
    __init__.py
    gomoku_game.py            # 五子棋游戏核心逻辑
    dataset_downloader.py     # 数据集下载工具
    evaluation.py            # 局面评估工具
  output/
    gomoku_dataset.json      # 下载的数据集
    *.json                   # 保存的游戏记录
```

### 安装依赖

建议使用虚拟环境：

```bash
cd week1314/Agent
pip install -r requirements.txt
```

### 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
QWEN_API_KEY=你的_qwen_api_key
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen-max
```

注意：不要将真实密钥写入代码库。推荐使用 `.env` 或系统环境变量。

### 使用方法

#### 1. AI 自主下棋（完整对局）

AI 将作为黑棋和白棋，自主完成一整局游戏：

```bash
python -m Agent.run_demo --mode auto
```

或直接运行：

```bash
cd week1314/Agent
python run_demo.py --mode auto
```

#### 2. AI vs 人类对局

人类先手，AI 作为对手：

```bash
python run_demo.py --mode human
```

#### 3. 自定义使用

在 Python 代码中使用：

```python
from agent_builder import build_agent

agent = build_agent()

# 初始化游戏
result = agent.invoke({"input": "请初始化一个15x15的五子棋游戏"})

# AI自主下棋
result = agent.invoke({
    "input": "请作为黑棋，查看棋盘状态，评估局面，然后走出最佳一步"
})
```

### Agent 工具集

Agent 可使用的工具包括：

1. **initGame**: 初始化新游戏
2. **getBoardState**: 查看当前棋盘状态（可视化）
3. **evaluatePosition**: 评估当前局面优劣
4. **suggestMoves**: 获取最佳走法建议（防守优先、攻击机会、优先级）
5. **makeMove**: 执行走子（格式：'row,col'）
6. **downloadDataset**: 下载五子棋数据集（棋谱、开局库）
7. **loadDataset**: 加载并查看数据集信息
8. **analyzeOpening**: 分析开局模式和走法统计
9. **saveGame**: 保存当前游戏状态
10. **loadGame**: 加载之前的游戏状态
11. **resetGame**: 重置游戏，重新开始

### 数据集功能

Agent 支持在线下载五子棋数据集：

- **棋谱数据集**: 包含经典对局记录
- **开局库**: 常见开局模式
- **自动下载**: 如果网络下载失败，会自动创建示例数据集

示例：

```python
# 下载数据集
agent.invoke({
    "input": "请下载五子棋数据集并保存到 data/games.json，类型为 games"
})

# 分析开局
agent.invoke({
    "input": "请分析 data/games.json 中的开局模式"
})
```

### 下棋策略

Agent 的下棋策略包括：

1. **防守优先**: 优先阻止对方形成五连
2. **攻击机会**: 寻找形成威胁的机会
3. **中心位置**: 优先占据中心区域
4. **局面评估**: 实时评估威胁和机会
5. **自主思考**: 基于 ReAct 框架的推理-行动循环

### 输出说明

运行后会生成：

- `output/gomoku_dataset.json`: 下载或创建的棋谱数据集
- `output/*.json`: 保存的游戏记录（可后续加载）

### 可自定义项

- 修改 `QWEN_MODEL` 使用不同模型（例如 `qwen-turbo`）
- 在 `tools/` 目录添加新工具，并在 `agent_builder.py` 中注册
- 调整 `temperature` 参数控制策略随机性（在 `agent_builder.py` 中）
- 修改棋盘大小（默认 15x15）在 `gomoku_game.py` 中

### 注意事项

1. 确保已配置 `QWEN_API_KEY` 环境变量
2. 网络连接正常（用于下载数据集）
3. 棋盘坐标范围为 0-14（15x15 棋盘）
4. 五连即获胜（横、竖、斜任意方向）
5. 黑棋先行

### 项目特色

- ✅ **完整的 ReAct 实现**: 思考-行动-观察循环
- ✅ **自主决策**: AI 能够独立思考和下棋
- ✅ **数据集学习**: 支持在线下载和学习开局模式
- ✅ **智能评估**: 局面分析和最佳走法推荐
- ✅ **游戏管理**: 保存、加载、重置功能完善

### 示例输出

运行 `python run_demo.py --mode auto` 后，你会看到：

```
=== Agent 思考过程 ===
Thought: 我需要初始化游戏，然后开始下棋...
Action: initGame
Action Input: 15
Observation: 已初始化 15x15 五子棋游戏，黑棋先行
...

=== Agent 最终答案 ===
我已经完成了10步棋的对局。作为黑棋，我采取了中心开局的策略...
```

### 故障排查

如果遇到问题：

1. **API 错误**: 检查 `.env` 文件中的 `QWEN_API_KEY` 是否正确
2. **网络错误**: 检查网络连接，数据集下载失败会自动创建示例数据
3. **导入错误**: 确保已安装所有依赖：`pip install -r requirements.txt`
4. **编码错误**: Windows 系统确保控制台支持 UTF-8
