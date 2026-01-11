# 使用Python官方镜像作为基础镜像
# 使用已手动拉取的镜像（python:3.10-slim）
# 终端运行： docker pull python:3.10-slim
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    DOCKER_ENV=1

# 安装系统依赖（用于matplotlib中文字体支持和 scikit-learn 编译）
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    fonts-wqy-zenhei \
    fonts-wqy-microhei \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 升级 pip 并安装 Python 依赖
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    python -c "import sklearn; print(f'scikit-learn version: {sklearn.__version__}')" && \
    python -c "from sklearn.decomposition import PCA; print('PCA import successful')"

# 复制应用代码
COPY XGBoost2pre.py .

# 创建必要的输出目录
RUN mkdir -p /app/output/{saved_models,preprocessors,feature_correlations,feature_importance,learning_curves,train_metrics,test_metrics,test_results} \
    && mkdir -p /app/dataset \
    && mkdir -p /app/validate_ds

# 设置matplotlib使用中文字体
RUN python -c "import matplotlib; print(matplotlib.get_cachedir())" && \
    python -c "import matplotlib.pyplot as plt; plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans']"

# 默认命令
CMD ["python", "XGBoost2pre.py", "--help"]

