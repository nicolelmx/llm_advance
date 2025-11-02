"""
五子棋游戏核心逻辑模块
提供棋盘管理、走子、胜负判断等功能
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from enum import Enum
import json
from pathlib import Path


class Player(Enum):
    """玩家枚举"""
    EMPTY = 0
    BLACK = 1  # 黑棋先手
    WHITE = 2  # 白棋后手


class GomokuBoard:
    """五子棋棋盘类"""
    
    def __init__(self, size: int = 15):
        """初始化棋盘
        
        Args:
            size: 棋盘大小，默认15x15（标准五子棋）
        """
        self.size = size
        self.board = [[Player.EMPTY for _ in range(size)] for _ in range(size)]
        self.current_player = Player.BLACK
        self.move_history: List[Tuple[int, int]] = []
        self.game_over = False
        self.winner: Optional[Player] = None
        
    def reset(self) -> None:
        """重置棋盘"""
        self.board = [[Player.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.current_player = Player.BLACK
        self.move_history = []
        self.game_over = False
        self.winner = None
        
    def is_valid_move(self, row: int, col: int) -> bool:
        """检查走子是否有效
        
        Args:
            row: 行坐标 (0-based)
            col: 列坐标 (0-based)
            
        Returns:
            是否有效
        """
        if self.game_over:
            return False
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        return self.board[row][col] == Player.EMPTY
    
    def make_move(self, row: int, col: int) -> bool:
        """执行走子
        
        Args:
            row: 行坐标
            col: 列坐标
            
        Returns:
            是否成功
        """
        if not self.is_valid_move(row, col):
            return False
            
        self.board[row][col] = self.current_player
        self.move_history.append((row, col))
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        else:
            # 切换玩家
            self.current_player = Player.WHITE if self.current_player == Player.BLACK else Player.BLACK
            
        return True
    
    def _check_win(self, row: int, col: int) -> bool:
        """检查当前位置是否形成五连（或以上）
        
        规则：自由五子（Freestyle）规则
        - 连续5颗同色棋子 = 获胜
        - 6连及以上也视为胜
        
        Args:
            row: 最后落子的行
            col: 最后落子的列
            
        Returns:
            是否获胜（形成5连或以上）
        """
        player = self.board[row][col]
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1),  # 副对角线
        ]
        
        for dr, dc in directions:
            count = 1  # 当前位置
            
            # 正向计数（检查最多10个位置，确保能检测6连及以上）
            for i in range(1, 10):
                nr, nc = row + dr * i, col + dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
                    
            # 反向计数（检查最多10个位置）
            for i in range(1, 10):
                nr, nc = row - dr * i, col - dc * i
                if 0 <= nr < self.size and 0 <= nc < self.size and self.board[nr][nc] == player:
                    count += 1
                else:
                    break
                    
            # 自由五子规则：5连或以上都算胜
            if count >= 5:
                return True
                
        return False
    
    def get_board_state(self) -> str:
        """获取棋盘状态的字符串表示（用于展示给LLM）"""
        lines = []
        lines.append(f"当前玩家: {'黑棋' if self.current_player == Player.BLACK else '白棋'}")
        lines.append(f"已走步数: {len(self.move_history)}")
        lines.append(f"棋盘上棋子总数: {len(self.move_history)} 个")
        if self.game_over:
            lines.append(f"\n【游戏结束】{'黑棋' if self.winner == Player.BLACK else '白棋'}获胜！")
            lines.append(f"【五连达成】{'黑棋' if self.winner == Player.BLACK else '白棋'}形成连续五颗棋子相连！")
        lines.append("")
        lines.append("棋盘状态（坐标从0开始，左上角为(0,0)）：")
        lines.append("   " + " ".join([f"{i%10}" for i in range(self.size)]))
        
        for i in range(self.size):
            row_str = f"{i:2} "
            for j in range(self.size):
                if self.board[i][j] == Player.BLACK:
                    row_str += "● "
                elif self.board[i][j] == Player.WHITE:
                    row_str += "○ "
                else:
                    row_str += ". "
            lines.append(row_str)
            
        if self.move_history:
            lines.append("")
            lines.append("走子历史:")
            for idx, (r, c) in enumerate(self.move_history):
                player_name = "黑棋" if idx % 2 == 0 else "白棋"
                lines.append(f"  {idx + 1}. {player_name}: ({r}, {c})")
                
        return "\n".join(lines)
    
    def get_board_json(self) -> dict:
        """获取棋盘状态的JSON表示"""
        board_data = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                if self.board[i][j] == Player.BLACK:
                    row.append(1)
                elif self.board[i][j] == Player.WHITE:
                    row.append(2)
                else:
                    row.append(0)
            board_data.append(row)
            
        return {
            "size": self.size,
            "board": board_data,
            "current_player": self.current_player.value,
            "move_history": self.move_history,
            "game_over": self.game_over,
            "winner": self.winner.value if self.winner else None,
        }
    
    def load_from_json(self, data: dict) -> bool:
        """从JSON加载棋盘状态"""
        try:
            self.size = data.get("size", 15)
            self.board = [[Player.EMPTY for _ in range(self.size)] for _ in range(self.size)]
            
            board_data = data.get("board", [])
            for i, row in enumerate(board_data):
                for j, val in enumerate(row):
                    if val == 1:
                        self.board[i][j] = Player.BLACK
                    elif val == 2:
                        self.board[i][j] = Player.WHITE
                        
            self.current_player = Player(data.get("current_player", 1))
            self.move_history = [tuple(m) for m in data.get("move_history", [])]
            self.game_over = data.get("game_over", False)
            winner_val = data.get("winner")
            self.winner = Player(winner_val) if winner_val else None
            
            return True
        except Exception:
            return False


# 全局游戏实例
_game_instance: Optional[GomokuBoard] = None


def init_game(size: int = 15) -> str:
    """初始化新游戏
    
    Args:
        size: 棋盘大小，默认15
        
    Returns:
        初始化结果信息
    """
    global _game_instance
    if _game_instance is not None and len(_game_instance.move_history) > 0:
        return f"警告：游戏已经进行中（已有{len(_game_instance.move_history)}步），如需重新开始请先调用resetGame。当前棋盘有{len(_game_instance.move_history)}个棋子。"
    _game_instance = GomokuBoard(size)
    return f"已初始化 {size}x{size} 五子棋游戏，黑棋先行。当前棋盘上有0个棋子。"


def get_current_board() -> GomokuBoard:
    """获取当前游戏实例"""
    global _game_instance
    if _game_instance is None:
        _game_instance = GomokuBoard()
    return _game_instance


def make_move(row: int, col: int) -> str:
    """执行走子
    
    Args:
        row: 行坐标 (0-14)
        col: 列坐标 (0-14)
        
    Returns:
        走子结果信息
    """
    board = get_current_board()
    
    if board.game_over:
        return "游戏已结束，无法继续走子。请重新开始游戏。"
    
    # 保存走子前的玩家（用于返回信息）
    player_who_moved = board.current_player
    
    if not board.make_move(row, col):
        return f"无效走子：位置 ({row}, {col}) 已被占用或超出边界"
    
    # 确定刚才走子的玩家名称
    player_name = "黑棋" if player_who_moved == Player.BLACK else "白棋"
    result = f"{player_name}在位置 ({row}, {col}) 落子成功"
    
    if board.game_over:
        winner_name = "黑棋" if board.winner == Player.BLACK else "白棋"
        result += f"\n\n【重要】游戏结束！{winner_name}获胜！\n"
        result += f"【重要】{winner_name}形成五连！对局已完成！\n"
        result += "【重要】请立即给出最终答案，不要再继续走子。"
    else:
        next_player = "黑棋" if board.current_player == Player.BLACK else "白棋"
        result += f"\n下一步轮到 {next_player}"
        result += f"\n当前棋盘上共有 {len(board.move_history)} 个棋子"
        
    return result


def get_board_state() -> str:
    """获取当前棋盘状态"""
    board = get_current_board()
    return board.get_board_state()


def save_game(filepath: str) -> str:
    """保存当前游戏状态到文件
    
    Args:
        filepath: 保存路径
        
    Returns:
        保存结果信息
    """
    board = get_current_board()
    data = board.get_board_json()
    
    try:
        path = Path(filepath).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return f"游戏已保存到: {path}"
    except Exception as e:
        return f"保存失败: {e}"


def load_game(filepath: str) -> str:
    """从文件加载游戏状态
    
    Args:
        filepath: 文件路径
        
    Returns:
        加载结果信息
    """
    global _game_instance
    
    try:
        path = Path(filepath).expanduser().resolve()
        if not path.exists():
            return f"文件不存在: {path}"
            
        data = json.loads(path.read_text(encoding="utf-8"))
        _game_instance = GomokuBoard()
        if _game_instance.load_from_json(data):
            return f"游戏已从 {path} 加载"
        else:
            return "加载失败：文件格式错误"
    except Exception as e:
        return f"加载失败: {e}"


def reset_game() -> str:
    """重置游戏（会清空棋盘上的所有棋子）"""
    board = get_current_board()
    previous_moves = len(board.move_history)
    board.reset()
    if previous_moves > 0:
        return f"警告：游戏已重置，清空了棋盘上的{previous_moves}个棋子。现在棋盘上有0个棋子，黑棋先行。"
    return "游戏已重置，黑棋先行。当前棋盘上有0个棋子。"

