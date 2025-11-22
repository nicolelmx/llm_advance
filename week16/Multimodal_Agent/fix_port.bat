@echo off
echo 正在查找占用8000端口的进程...
netstat -ano | findstr :8000
echo.
echo 如果看到进程ID，可以使用以下命令关闭：
echo taskkill /PID [进程ID] /F
echo.
echo 或者修改 .env 文件中的 API_PORT 为其他端口（如 8001）
pause

