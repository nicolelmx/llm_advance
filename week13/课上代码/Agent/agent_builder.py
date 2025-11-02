"""
五子棋AI Agent构建模块
基于LangChain ReAct框架，构建自主下棋的智能体
"""

from __future__ import annotations

from typing import List

from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.caches import BaseCache  # noqa: F401
import httpx
from langchain.agents import create_react_agent, AgentExecutor

from config import load_config
from tools import (
    init_game,
    make_move,
    get_board_state,
    save_game,
    load_game,
    reset_game,
    download_gomoku_dataset,
    load_dataset,
    analyze_opening,
    evaluate_position,
    suggest_moves,
    analyze_pattern,
)


def _parse_and_make_move(pos_str: str) -> str:
    """解析位置字符串并执行走子"""
    try:
        # 支持多种格式: "7,7", "7 7", "row=7,col=7" 等
        pos_str = pos_str.strip()
        if ',' in pos_str:
            parts = pos_str.split(',')
        else:
            parts = pos_str.split()
            
        if len(parts) < 2:
            return "错误：需要提供行和列坐标，格式为 'row,col'，例如 '7,7'"
            
        row = int(parts[0].strip())
        col = int(parts[1].strip())
        return make_move(row, col)
    except ValueError as e:
        return f"错误：坐标格式无效，请使用 'row,col' 格式，例如 '7,7'。错误: {e}"
    except Exception as e:
        return f"走子失败: {e}"


def _parse_download_args(args: str) -> str:
    """解析下载参数"""
    try:
        args = args.strip()
        # 支持多种分隔符
        if ',' in args:
            parts = [p.strip() for p in args.split(',', 1)]
        elif ' ' in args:
            parts = args.split(maxsplit=1)
            if len(parts) == 1:
                parts.append('games')
        else:
            # 只有路径，默认games类型
            parts = [args, 'games']
            
        save_path = parts[0].strip().strip('"\'')
        dataset_type = parts[1].strip().strip('"\'').lower() if len(parts) > 1 else 'games'
        
        # 验证dataset_type
        if dataset_type not in ['games', 'openings']:
            dataset_type = 'games'
            
        return download_gomoku_dataset(save_path, dataset_type)
    except Exception as e:
        return f"下载失败: {e}"


def build_tools() -> List[Tool]:
    """构建五子棋Agent的工具集"""
    return [
        Tool(
            name="initGame",
            description=(
                "初始化新的五子棋游戏。"
                "输入: 棋盘大小（可选，默认15），例如 '15' 或 'size=15'"
                "返回: 初始化结果信息"
            ),
            func=lambda size_str: init_game(int(size_str.strip()) if size_str.strip().isdigit() else 15),
        ),
        Tool(
            name="makeMove",
            description=(
                "在棋盘上落子。"
                "输入: 位置坐标，格式为 'row,col' 或 'row col'，例如 '7,7' 或 '7 7'"
                "坐标范围: 0-14（15x15棋盘）"
                "返回: 走子结果，包括是否成功、下一步玩家、游戏状态等"
            ),
            func=lambda pos: _parse_and_make_move(pos),
        ),
        Tool(
            name="getBoardState",
            description=(
                "获取当前棋盘状态的完整信息。"
                "输入: 任意文本（通常为 'current' 或 'state'）"
                "返回: 棋盘可视化、当前玩家、走子历史、游戏状态等详细信息"
            ),
            func=lambda _: get_board_state(),
        ),
        Tool(
            name="evaluatePosition",
            description=(
                "评估当前局面的优劣。"
                "输入: 任意文本（通常为 'evaluate' 或 'analyze'）"
                "返回: 局面评估，包括威胁点、机会点、当前状态等"
            ),
            func=lambda _: evaluate_position(),
        ),
        Tool(
            name="suggestMoves",
            description=(
                "获取最佳走法建议。"
                "输入: 可选的最大建议数量（默认为5），例如 '5'"
                "返回: 按照优先级排序的走法建议列表，包括位置、原因、优先级评分"
            ),
            func=lambda n: suggest_moves(int(n.strip()) if n.strip().isdigit() else 5),
        ),
        Tool(
            name="downloadDataset",
            description=(
                "下载五子棋数据集（棋谱、开局库等）。"
                "输入: 保存路径和dataset_type，格式必须为 'save_path,dataset_type'"
                "示例: 'output/gomoku_dataset.json,games' 或 'data/openings.json,openings'"
                "dataset_type必须是: 'games' 或 'openings'"
                "返回: 下载结果，包括保存路径和数据统计"
            ),
            func=_parse_download_args,
        ),
        Tool(
            name="loadDataset",
            description=(
                "加载并查看数据集信息。"
                "输入: 数据集文件路径"
                "返回: 数据集摘要信息，包括条目数量、格式等"
            ),
            func=load_dataset,
        ),
        Tool(
            name="analyzeOpening",
            description=(
                "分析开局模式和走法统计。"
                "输入: 数据集文件路径"
                "返回: 开局走法统计，包括常见第一步走法等"
            ),
            func=analyze_opening,
        ),
        Tool(
            name="saveGame",
            description=(
                "保存当前游戏状态到文件。"
                "输入: 保存文件路径，例如 'games/game1.json'"
                "返回: 保存结果和文件路径"
            ),
            func=save_game,
        ),
        Tool(
            name="loadGame",
            description=(
                "从文件加载游戏状态。"
                "输入: 游戏文件路径"
                "返回: 加载结果信息"
            ),
            func=load_game,
        ),
        Tool(
            name="resetGame",
            description=(
                "重置当前游戏，清空棋盘，黑棋先行。"
                "输入: 任意文本（通常为 'reset'）"
                "返回: 重置确认信息"
            ),
            func=lambda _: reset_game(),
        ),
    ]


def build_agent() -> AgentExecutor:
    """构建五子棋AI Agent"""
    # 兼容部分环境下 Pydantic 类未完全构建的问题
    try:
        ChatOpenAI.model_rebuild(force=True)
    except Exception:
        pass
    
    cfg = load_config()

    http_client = httpx.Client(timeout=60.0, http2=False)

    llm = ChatOpenAI(
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        model=cfg["model"],
        temperature=0.3,  # 降低温度以保持策略稳定性
        max_retries=2,
        http_client=http_client,
    )

    tools = build_tools()

    # 使用标准的 ReAct prompt，必须使用英文关键词以确保 LangChain 正确解析
    prompt = PromptTemplate.from_template(
        """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

**重要规则：**
1. Action 和 Action Input 必须分开写，格式如下：
   Action: tool_name
   Action Input: input_value
   
2. 不要写成 Action: tool_name('input') 这样的格式，这是错误的！

3. 一旦任务完成，立即给出 Final Answer，不要再调用工具

4. 对于简单任务（如初始化游戏），调用工具后直接给出 Final Answer

**示例格式：**
Question: 初始化一个15x15的五子棋游戏
Thought: 我需要调用 initGame 工具来初始化游戏
Action: initGame
Action Input: 15
Observation: 已初始化 15x15 五子棋游戏，黑棋先行
Thought: 游戏已成功初始化，任务完成
Final Answer: 游戏已成功初始化为15x15大小，黑棋先行

**工具说明：**
- initGame: 初始化新游戏，输入为棋盘大小（如 '15'）
- getBoardState: 查看当前棋盘状态，输入任意文本（如 'current'）
- makeMove: 执行走子，输入格式为 'row,col'（如 '7,7'）
- evaluatePosition: 评估当前局面，输入任意文本
- suggestMoves: 获取最佳走法建议，输入建议数量（如 '5'）
- downloadDataset: 下载数据集，输入格式为 'path,type'（如 'path.json,games'）
- 其他工具请参考工具描述

Question: {input}
Thought: {agent_scratchpad}
        """.strip()
    )

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    
    # 自定义解析错误处理函数
    def handle_parsing_error(error: Exception) -> str:
        """处理解析错误，返回友好的错误信息"""
        return f"解析错误，请严格按照格式输出。错误: {str(error)[:100]}"
    
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,  # 关闭详细输出，减少噪音
        max_iterations=50,  # 增加迭代次数，确保对局能完成（最多可能需要50步才能分出胜负）
        handle_parsing_errors=handle_parsing_error,  # 使用自定义错误处理
        return_intermediate_steps=True,  # 返回中间步骤以便调试
        max_execution_time=600,  # 设置最大执行时间10分钟
        early_stopping_method="force",  # 强制停止方法
    )
    return executor
