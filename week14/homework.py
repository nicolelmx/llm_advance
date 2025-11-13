"""
📚第14周作业要求：实现以下功能（二选一）
1️⃣:如果要将项目拓展为支持多人对战，需要做那些改动？
状态管理：'_game_instance'改成映射为游戏ID，进一步添加房间管理
网络管理：添加webSocket或者http api接口，实现消息队列，进一部分处理玩家操作
并发：使用异步框架（Fast api， asyncio）
需要使用向量数据库，存储游戏状态和历史记录
玩家的合法性检查和身份验证

2️⃣：添加难度级别AI系统
beginner:初级，随机走法，偶尔防守
intermediate：中级，（BFS DFS），deep = 2
Advance：高级，deep = 4
Expert：建议使用网络已有的库，深度训练（deepmind-AlphaGo）

"""