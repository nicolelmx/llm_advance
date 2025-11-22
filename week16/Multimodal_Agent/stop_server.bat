@echo off
echo ============================================================
echo   停止多模态数据分析师Agent服务器
echo ============================================================
echo.

REM 查找占用8000端口的进程
echo 正在查找占用端口8000的进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo 找到进程ID: %%a
    echo 正在关闭进程...
    taskkill /PID %%a /F >nul 2>&1
    if errorlevel 1 (
        echo [错误] 无法关闭进程 %%a
    ) else (
        echo [成功] 已关闭进程 %%a
    )
    goto :found
)

echo [提示] 未找到占用端口8000的进程
:found

echo.
echo ============================================================
pause

