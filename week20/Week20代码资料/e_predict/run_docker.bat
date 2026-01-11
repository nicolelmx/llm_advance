@echo off
REM Docker快速启动脚本 (Windows)

echo ========================================
echo XGBoost 预测模型 Docker 启动脚本
echo ========================================

REM 检查Docker是否安装
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Docker，请先安装Docker Desktop
    pause
    exit /b 1
)

REM 检查docker-compose是否安装
where docker-compose >nul 2>&1
if %errorlevel% neq 0 (
    echo 警告: 未找到docker-compose，将使用docker命令
    set USE_COMPOSE=false
) else (
    set USE_COMPOSE=true
)

REM 构建镜像
echo 正在构建Docker镜像...
if "%USE_COMPOSE%"=="true" (
    docker-compose build
) else (
    docker build -t xgboost-predict .
)

if %errorlevel% neq 0 (
    echo 镜像构建失败
    pause
    exit /b 1
)

REM 显示使用说明
echo ========================================
echo 镜像构建完成！
echo ========================================
echo.
echo 使用示例：
echo.
echo 1. 基础训练：
if "%USE_COMPOSE%"=="true" (
    echo    docker-compose run --rm xgboost-train python XGBoost2pre.py --normalize
) else (
    echo    docker run --rm -v "%cd%:/app" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --normalize
)
echo.
echo 2. 使用PCA降维：
if "%USE_COMPOSE%"=="true" (
    echo    docker-compose run --rm xgboost-train python XGBoost2pre.py --use_pca --n_components 10 --normalize
) else (
    echo    docker run --rm -v "%cd%:/app" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --use_pca --n_components 10 --normalize
)
echo.
echo 3. 查看帮助：
if "%USE_COMPOSE%"=="true" (
    echo    docker-compose run --rm xgboost-train python XGBoost2pre.py --help
) else (
    echo    docker run --rm -v "%cd%:/app" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --help
)
echo.
echo 更多信息请查看 DOCKER_USAGE.md
pause

