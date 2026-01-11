#!/bin/bash
# Docker快速启动脚本

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}XGBoost 预测模型 Docker 启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}错误: 未找到Docker，请先安装Docker${NC}"
    exit 1
fi

# 检查docker-compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}警告: 未找到docker-compose，将使用docker命令${NC}"
    USE_COMPOSE=false
else
    USE_COMPOSE=true
fi

# 构建镜像
echo -e "${GREEN}正在构建Docker镜像...${NC}"
if [ "$USE_COMPOSE" = true ]; then
    docker-compose build
else
    docker build -t xgboost-predict .
fi

if [ $? -ne 0 ]; then
    echo -e "${YELLOW}镜像构建失败${NC}"
    exit 1
fi

# 显示使用说明
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}镜像构建完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "使用示例："
echo ""
echo "1. 基础训练："
if [ "$USE_COMPOSE" = true ]; then
    echo "   docker-compose run --rm xgboost-train python XGBoost2pre.py --normalize"
else
    echo "   docker run --rm -v \"\$(pwd):/app\" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --normalize"
fi
echo ""
echo "2. 使用PCA降维："
if [ "$USE_COMPOSE" = true ]; then
    echo "   docker-compose run --rm xgboost-train python XGBoost2pre.py --use_pca --n_components 10 --normalize"
else
    echo "   docker run --rm -v \"\$(pwd):/app\" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --use_pca --n_components 10 --normalize"
fi
echo ""
echo "3. 查看帮助："
if [ "$USE_COMPOSE" = true ]; then
    echo "   docker-compose run --rm xgboost-train python XGBoost2pre.py --help"
else
    echo "   docker run --rm -v \"\$(pwd):/app\" -e DOCKER_ENV=1 xgboost-predict python XGBoost2pre.py --help"
fi
echo ""
echo "更多信息请查看 DOCKER_USAGE.md"

