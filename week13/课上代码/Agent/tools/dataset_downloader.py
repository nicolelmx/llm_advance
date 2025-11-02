"""
数据集下载工具
支持在线下载五子棋棋谱数据集
"""

from __future__ import annotations

from pathlib import Path
import json
import urllib.request
import urllib.error
from typing import List, Dict, Optional
import socket


def download_gomoku_dataset(
    save_path: str,
    dataset_type: str = "games"
) -> str:
    """下载五子棋数据集
    
    Args:
        save_path: 保存路径
        dataset_type: 数据集类型 (games, openings, patterns)
        
    Returns:
        下载结果信息
    """
    # 预设的数据集URL（示例）
    preset_urls = {
        "games": "https://raw.githubusercontent.com/example/gomoku-games/main/sample_games.json",
        "openings": "https://raw.githubusercontent.com/example/gomoku-games/main/opening_book.json",
    }
    
    # 使用预设URL或尝试从dataset_type构建URL
    actual_url = preset_urls.get(dataset_type, preset_urls.get("games"))
    
    try:
        path = Path(save_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 尝试下载文件（添加超时，避免卡住）
        try:
            # 设置socket超时为5秒
            socket.setdefaulttimeout(5)
            print(f"正在从 {actual_url} 下载数据...")
            urllib.request.urlretrieve(actual_url, path)
            
            # 验证是否为有效JSON
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                game_count = len(data) if isinstance(data, list) else 1
                return f"下载成功！已保存到: {path}\n数据集包含 {game_count} 个条目"
            except json.JSONDecodeError:
                # 如果不是JSON，可能是文本格式的棋谱
                return f"下载成功！已保存到: {path}\n（文件格式：文本）"
                
        except (urllib.error.URLError, socket.timeout, TimeoutError):
            # URL无效或超时，直接创建示例数据集
            return _create_sample_dataset(save_path, dataset_type)
            
    except Exception as e:
        # 其他异常，也创建示例数据集
        return _create_sample_dataset(save_path, dataset_type)


def _create_sample_dataset(save_path: str, dataset_type: str) -> str:
    """创建示例数据集（当无法下载时）"""
    path = Path(save_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if dataset_type == "games":
        # 示例游戏棋谱
        sample_games = [
            {
                "id": 1,
                "moves": [(7, 7), (8, 8), (7, 8), (8, 7), (7, 9), (8, 6)],
                "winner": "black",
                "result": "black_wins",
            },
            {
                "id": 2,
                "moves": [(7, 7), (7, 8), (8, 8), (8, 7), (9, 8), (6, 8)],
                "winner": "white",
                "result": "white_wins",
            },
            {
                "id": 3,
                "moves": [
                    (7, 7), (7, 8), (8, 7), (8, 8), (9, 7), (6, 8),
                    (7, 9), (8, 9), (9, 8), (10, 7), (7, 10)
                ],
                "winner": "black",
                "result": "black_wins",
            }
        ]
        
        path.write_text(
            json.dumps(sample_games, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return f"创建示例数据集成功！已保存到: {path}\n包含 {len(sample_games)} 局示例棋谱"
        
    elif dataset_type == "openings":
        # 开局库
        openings = [
            {
                "name": "天元开局",
                "first_move": (7, 7),
                "follow_up": [(8, 8), (7, 8), (8, 7)],
            },
            {
                "name": "侧翼开局",
                "first_move": (7, 6),
                "follow_up": [(8, 7), (7, 7), (8, 8)],
            },
        ]
        
        path.write_text(
            json.dumps(openings, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
        return f"创建开局库成功！已保存到: {path}\n包含 {len(openings)} 种开局"
    else:
        # 默认创建游戏数据集
        return _create_sample_dataset(save_path, "games")


def load_dataset(filepath: str) -> str:
    """加载数据集并返回摘要信息
    
    Args:
        filepath: 数据集文件路径
        
    Returns:
        数据集摘要信息
    """
    try:
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            return f"文件不存在: {path}"
            
        data = json.loads(path.read_text(encoding="utf-8"))
        
        if isinstance(data, list):
            count = len(data)
            if count > 0:
                first_item = data[0]
                summary = f"数据集包含 {count} 个条目\n"
                summary += f"示例条目键: {list(first_item.keys()) if isinstance(first_item, dict) else '列表项'}"
                return summary
            else:
                return "数据集为空"
        else:
            return f"数据集格式: {type(data).__name__}\n键: {list(data.keys()) if isinstance(data, dict) else 'N/A'}"
            
    except json.JSONDecodeError:
        return f"文件不是有效的JSON格式: {filepath}"
    except Exception as e:
        return f"加载失败: {e}"


def analyze_opening(dataset_path: str, opening_name: Optional[str] = None) -> str:
    """分析开局模式
    
    Args:
        dataset_path: 数据集路径
        opening_name: 开局名称（可选）
        
    Returns:
        开局分析结果
    """
    try:
        path = Path(dataset_path).expanduser().resolve()
        if not path.exists():
            return f"文件不存在: {path}"
            
        data = json.loads(path.read_text(encoding="utf-8"))
        
        if not isinstance(data, list):
            return "数据集格式不支持开局分析"
            
        # 统计第一步走法
        first_moves = {}
        for game in data:
            if isinstance(game, dict) and "moves" in game:
                moves = game.get("moves", [])
                if moves:
                    first_move = moves[0]
                    first_moves[first_move] = first_moves.get(first_move, 0) + 1
                    
        if first_moves:
            result = "开局走法统计:\n"
            sorted_moves = sorted(first_moves.items(), key=lambda x: x[1], reverse=True)
            for move, count in sorted_moves[:10]:  # 显示前10个
                result += f"  {move}: {count} 次\n"
            return result.strip()
        else:
            return "无法从数据集中提取开局信息"
            
    except Exception as e:
        return f"分析失败: {e}"

