# Docker 操作指南 - ProjectA & ProjectB

## 概述

- **ProjectA**：包含完整的 Docker 部署配置（Dockerfile + docker-compose.yml）
- **ProjectB**：目前没有 Docker 配置，仅支持本地部署

---

## ProjectA Docker 操作步骤

### 一、前置准备

#### 1.1 安装 Docker

**Windows 系统**（两种方式）：

**方式1：使用 Docker Desktop（推荐）**
- 下载并安装 [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop)
- Docker Desktop 会自动配置 WSL 2 后端（如果已安装 WSL）
- 安装完成后重启电脑
- 启动 Docker Desktop 应用

**方式2：在 WSL 中直接安装 Docker Engine**
- 如果已安装 WSL（如 Ubuntu），可以在 WSL 中直接安装 Docker Engine
- 这种方式不需要 Docker Desktop，但需要手动配置

**WSL 安装和配置**（如果还没有安装 WSL）：

```powershell
# 1. 检查 WSL 是否已安装
wsl --status

# 2. 如果未安装，安装 WSL
wsl --install

# 3. 查看可用的 Linux 发行版
wsl --list --online

# 4. 安装 Ubuntu（推荐，最常用）
wsl --install -d Ubuntu

# 或安装 Ubuntu 22.04 LTS（长期支持版本）
wsl --install -d Ubuntu-22.04

# 5. 安装完成后，设置 WSL 版本为 2（推荐）
wsl --set-default-version 2

# 6. 查看已安装的发行版
wsl --list --verbose
```

**验证安装**：
```powershell
# 检查 Docker 版本（在 PowerShell 中）
docker --version

# 检查 Docker Compose 版本（可选）
docker-compose --version

# 检查 Docker 服务是否运行
docker ps

# 如果使用 WSL，也可以在 WSL 中检查
wsl docker --version
```

#### 1.2 准备项目文件

在构建 Docker 镜像之前，确保以下文件已准备好：

```bash
week20/ProjectA/
├── models/                    # ✅ 必须：训练好的模型文件
│   ├── lgbm_model.pkl        # 或 lgbm_model.onnx
│   └── lgbm_model.txt
├── data/processed/           # ✅ 必须：预处理文件
│   ├── scaler.joblib
│   └── columns.json
├── src/                      # ✅ 必须：源代码
│   ├── api/
│   │   └── app.py
│   └── ...
├── requirements.txt          # ✅ 必须：依赖列表
├── Dockerfile                # ✅ 必须：Docker 构建文件
└── docker-compose.yml        # ✅ 可选：Docker Compose 配置
```

**重要**：如果还没有训练模型，请先执行以下步骤：

```powershell
# 1. 进入项目目录
cd week20\ProjectA

# 2. 数据预处理
python src\preprocess.py

# 3. 训练模型
python src\train_lgbm.py
```

---

### 二、Docker 镜像构建

#### 2.1 基本构建命令

在项目根目录（`week20/ProjectA/`）执行：

```powershell
# Windows PowerShell
cd week20\ProjectA
docker build -t risk-api:latest .
```

**参数说明**：
- `-t risk-api:latest`：指定镜像名称和标签（tag）
- `.`：使用当前目录作为构建上下文（包含 Dockerfile 的目录）

#### 2.2 构建过程说明

构建过程会执行以下步骤：

1. **下载基础镜像**：`python:3.10-slim`（首次构建需要下载，约 100-200MB）
2. **安装系统依赖**：curl 等工具
3. **安装 Python 依赖**：根据 `requirements.txt` 安装所有包
4. **复制项目文件**：
   - `src/` → `/app/src`
   - `models/` → `/app/models`
   - `data/processed/` → `/app/data/processed`
5. **设置环境变量**：PORT=8000
6. **暴露端口**：8000

**构建时间**：
- 首次构建：5-10 分钟（需要下载基础镜像和依赖）
- 后续构建：1-3 分钟（使用缓存）

#### 2.3 构建输出示例

```
[+] Building 45.2s (10/10) FINISHED
 => [internal] load build definition from Dockerfile
 => => transferring dockerfile: 2.00kB
 => [1/6] FROM docker.io/library/python:3.10-slim
 => [2/6] RUN apt-get update && apt-get install -y ...
 => [3/6] COPY requirements.txt .
 => [4/6] RUN pip install --no-cache-dir -r requirements.txt
 => [5/6] COPY src ./src
 => [6/6] COPY models ./models
 => [7/6] COPY data/processed ./data/processed
 => exporting to image
 => => exporting layers
 => => writing image sha256:...
 => => naming to docker.io/library/risk-api:latest
```

#### 2.4 验证镜像构建成功

```powershell
# 查看本地镜像列表
docker images

# 应该看到 risk-api:latest
# REPOSITORY    TAG       IMAGE ID       CREATED         SIZE
# risk-api      latest    abc123def456   2 minutes ago   500MB
```

---

### 三、Docker 容器运行

#### 3.1 基本运行方式

**方式1：后台运行（推荐）**
```powershell
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest
```

**参数说明**：
- `-d`：后台运行（detached mode），容器在后台运行
- `-p 8000:8000`：端口映射（主机端口:容器端口）
  - 主机端口 8000 → 容器端口 8000
- `--name risk-api-container`：指定容器名称
- `risk-api:latest`：使用的镜像名称

**方式2：交互式运行（查看实时日志）**
```powershell
docker run -it -p 8000:8000 --name risk-api-container risk-api:latest
```

**参数说明**：
- `-it`：交互式运行，可以看到实时日志输出
- 按 `Ctrl+C` 停止容器

**方式3：自定义端口**
```powershell
# 如果主机 8000 端口被占用，使用其他端口
docker run -d -p 8080:8000 --name risk-api-container risk-api:latest
```

访问地址：`http://localhost:8080`

#### 3.2 验证容器运行

**检查容器状态**：
```powershell
# 查看运行中的容器
docker ps

# 输出示例：
# CONTAINER ID   IMAGE            COMMAND                  STATUS         PORTS                    NAMES
# abc123def456   risk-api:latest  "uvicorn src.api..."     Up 2 minutes   0.0.0.0:8000->8000/tcp   risk-api-container

# 查看所有容器（包括已停止的）
docker ps -a
```

**查看容器日志**：
```powershell
# 查看日志
docker logs risk-api-container

# 实时查看日志（类似 tail -f）
docker logs -f risk-api-container
```

**测试 API**：
```powershell
# 健康检查
curl http://localhost:8000/health

# 预期输出：{"status":"ok"}

# 查看 API 文档（浏览器访问）
# http://localhost:8000/docs
```

---

### 四、使用 Docker Compose（推荐）

Docker Compose 可以更方便地管理容器，支持配置文件、健康检查、自动重启等功能。

#### 4.1 使用 Docker Compose 启动

```powershell
# 进入项目目录
cd week20\ProjectA

# 构建并启动（后台运行）
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

#### 4.2 Docker Compose 命令说明

```powershell
# 启动服务（后台运行）
docker-compose up -d

# 启动服务（前台运行，查看日志）
docker-compose up

# 停止服务
docker-compose down

# 停止并删除容器和网络（保留镜像）
docker-compose down

# 停止并删除容器、网络、卷（完全清理）
docker-compose down -v

# 重新构建镜像并启动
docker-compose up -d --build

# 查看日志
docker-compose logs

# 实时查看日志
docker-compose logs -f

# 查看服务状态
docker-compose ps

# 重启服务
docker-compose restart
```

#### 4.3 docker-compose.yml 配置说明

```yaml
version: '3.8'

services:
  risk-api:
    build:
      context: .              # 构建上下文（当前目录）
      dockerfile: Dockerfile  # Dockerfile 路径
    image: risk-api:latest     # 镜像名称
    container_name: risk-api-container  # 容器名称
    ports:
      - "8000:8000"           # 端口映射
    environment:
      - PORT=8000             # 环境变量
    restart: unless-stopped   # 自动重启策略
    healthcheck:              # 健康检查
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s           # 检查间隔
      timeout: 10s            # 超时时间
      retries: 3              # 重试次数
      start_period: 40s       # 启动宽限期
```

---

### 五、容器管理命令

#### 5.1 基本管理命令

```powershell
# 停止容器
docker stop risk-api-container

# 启动已停止的容器
docker start risk-api-container

# 重启容器
docker restart risk-api-container

# 删除容器（必须先停止）
docker stop risk-api-container
docker rm risk-api-container

# 强制删除容器（即使正在运行）
docker rm -f risk-api-container
```

#### 5.2 进入容器内部（调试用）

```powershell
# 进入容器的 bash shell
docker exec -it risk-api-container /bin/bash

# 在容器内执行命令（不进入 shell）
docker exec risk-api-container ls /app

# 查看容器内的进程
docker exec risk-api-container ps aux
```

#### 5.3 查看容器资源使用

```powershell
# 查看容器资源使用情况（CPU、内存、网络等）
docker stats risk-api-container

# 查看所有容器的资源使用
docker stats

# 只查看一次（不持续更新）
docker stats --no-stream risk-api-container
```

#### 5.4 复制文件到/从容器

```powershell
# 从容器复制文件到主机
docker cp risk-api-container:/app/models/lgbm_model.pkl ./models/

# 从主机复制文件到容器
docker cp ./test.json risk-api-container:/app/
```

---

### 六、镜像管理

#### 6.1 查看镜像

```powershell
# 查看所有本地镜像
docker images

# 查看特定镜像
docker images risk-api

# 查看镜像详细信息
docker inspect risk-api:latest
```

#### 6.2 删除镜像

```powershell
# 先删除使用该镜像的容器
docker rm -f risk-api-container

# 删除镜像
docker rmi risk-api:latest

# 强制删除镜像（即使有容器使用）
docker rmi -f risk-api:latest
```

#### 6.3 导出和导入镜像

```powershell
# 导出镜像到文件
docker save -o risk-api.tar risk-api:latest

# 从文件导入镜像
docker load -i risk-api.tar
```

---

### 七、常见问题排查

#### 7.1 端口已被占用

**错误信息**：`Error: bind: address already in use`

**解决方法**：

```powershell
# 方法1：使用其他端口
docker run -d -p 8080:8000 --name risk-api-container risk-api:latest

# 方法2：查找并停止占用端口的进程（Windows）
netstat -ano | findstr :8000
# 找到 PID，然后结束进程
taskkill /PID <PID> /F

# 方法3：停止占用端口的容器
docker ps | findstr 8000
docker stop <container_id>
```

#### 7.2 模型文件未找到

**错误信息**：`FileNotFoundError: No model found (onnx or pkl).`

**解决方法**：

```powershell
# 1. 确保在构建镜像前已训练模型
cd week20\ProjectA
python src\train_lgbm.py

# 2. 检查 models/ 目录下是否有模型文件
dir models\

# 3. 重新构建镜像
docker build -t risk-api:latest .
```

#### 7.3 容器启动后立即退出

**诊断步骤**：

```powershell
# 1. 查看容器日志
docker logs risk-api-container

# 2. 检查容器退出代码
docker ps -a | findstr risk-api-container

# 3. 交互式运行查看错误
docker run -it -p 8000:8000 risk-api:latest
```

**常见原因**：
- 依赖安装失败
- 模型文件缺失
- 代码错误
- 端口冲突

#### 7.4 构建镜像时网络超时

**解决方法**：

```powershell
# 方法1：配置 Docker 镜像加速器（国内用户）
# 在 Docker Desktop 设置中添加镜像源：
# https://docker.mirrors.ustc.edu.cn
# https://registry.docker-cn.com

# 方法2：使用代理（如果有）
# 在 Docker Desktop 设置中配置代理

# 方法3：重试构建
docker build -t risk-api:latest . --no-cache
```

#### 7.5 容器内无法访问外部网络

**解决方法**：

```powershell
# 检查 Docker 网络设置
docker network ls

# 查看容器网络配置
docker inspect risk-api-container | findstr NetworkMode

# 使用 host 网络模式（Linux，Windows 不支持）
docker run -d --network host --name risk-api-container risk-api:latest
```

#### 7.6 容器内文件权限问题

**解决方法**：

```powershell
# 在 Dockerfile 中添加用户设置（如果需要）
# 或使用 --user 参数运行容器
docker run -d -p 8000:8000 --user $(id -u):$(id -g) --name risk-api-container risk-api:latest
```

---

### 八、生产环境建议

#### 8.1 资源限制

```powershell
# 限制内存和 CPU
docker run -d -p 8000:8000 \
  --memory="512m" \
  --cpus="1.0" \
  --name risk-api-container \
  risk-api:latest
```

#### 8.2 环境变量配置

```powershell
# 通过环境变量配置
docker run -d -p 8000:8000 \
  -e PORT=8000 \
  -e LOG_LEVEL=info \
  -e MODEL_PATH=/app/models/lgbm_model.pkl \
  --name risk-api-container \
  risk-api:latest
```

#### 8.3 数据持久化（Volume）

如果需要持久化数据或模型文件：

```powershell
# 挂载主机目录到容器
docker run -d -p 8000:8000 \
  -v E:\培训\week20\ProjectA\models:/app/models \
  -v E:\培训\week20\ProjectA\data:/app/data \
  --name risk-api-container \
  risk-api:latest
```

#### 8.4 使用 Docker Compose 配置资源限制

在 `docker-compose.yml` 中添加：

```yaml
services:
  risk-api:
    # ... 其他配置
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

---

### 九、完整部署流程示例

```powershell
# ========== 完整部署流程 ==========

# 1. 进入项目目录
cd E:\培训\week20\ProjectA

# 2. 确保模型已训练（如果还没有）
python src\preprocess.py
python src\train_lgbm.py

# 3. 构建 Docker 镜像
docker build -t risk-api:latest .

# 4. 查看镜像是否构建成功
docker images | findstr risk-api

# 5. 运行容器（后台运行）
docker run -d -p 8000:8000 --name risk-api-container risk-api:latest

# 6. 查看容器状态
docker ps | findstr risk-api-container

# 7. 查看日志确认启动成功
docker logs risk-api-container

# 8. 测试 API
curl http://localhost:8000/health

# 9. 访问 API 文档
# 浏览器打开：http://localhost:8000/docs

# 10. 停止容器（测试完成后）
docker stop risk-api-container

# 11. 删除容器（可选）
docker rm risk-api-container
```

---

### 十、使用 Docker Compose 的完整流程

```powershell
# ========== 使用 Docker Compose ==========

# 1. 进入项目目录
cd E:\培训\week20\ProjectA

# 2. 确保模型已训练
python src\preprocess.py
python src\train_lgbm.py

# 3. 构建并启动服务（后台运行）
docker-compose up -d

# 4. 查看服务状态
docker-compose ps

# 5. 查看日志
docker-compose logs -f

# 6. 测试 API
curl http://localhost:8000/health

# 7. 停止服务
docker-compose down

# 8. 重新构建并启动（代码更新后）
docker-compose up -d --build
```

---

## ProjectB Docker 部署（未来扩展）

ProjectB 目前没有 Docker 配置，如果需要 Docker 部署，可以参考 ProjectA 的配置，添加以下内容：

### 需要添加的文件

1. **Dockerfile**：基于 ProjectA 的 Dockerfile，添加 Ollama 客户端依赖
2. **docker-compose.yml**：可以同时启动 API 服务和 Ollama 服务

### 注意事项

- ProjectB 依赖 Ollama 服务，需要确保 Ollama 在容器内或外部可访问
- 如果 Ollama 在外部运行，需要配置网络连接
- 如果 Ollama 在容器内运行，需要更大的镜像和更多资源

---

## 总结

### ProjectA Docker 操作快速参考

| 操作 | 命令 |
|------|------|
| 构建镜像 | `docker build -t risk-api:latest .` |
| 运行容器 | `docker run -d -p 8000:8000 --name risk-api-container risk-api:latest` |
| 查看日志 | `docker logs -f risk-api-container` |
| 停止容器 | `docker stop risk-api-container` |
| 删除容器 | `docker rm risk-api-container` |
| 使用 Compose | `docker-compose up -d` |
| 停止 Compose | `docker-compose down` |

### 关键检查点

1. ✅ Docker 已安装并运行
2. ✅ 模型文件已训练（`models/lgbm_model.pkl`）
3. ✅ 预处理文件存在（`data/processed/`）
4. ✅ 端口 8000 未被占用
5. ✅ 镜像构建成功
6. ✅ 容器运行正常
7. ✅ API 健康检查通过

---

## 参考资源

- [Docker 官方文档](https://docs.docker.com/)
- [Docker Compose 文档](https://docs.docker.com/compose/)
- [FastAPI 部署文档](https://fastapi.tiangolo.com/deployment/)

