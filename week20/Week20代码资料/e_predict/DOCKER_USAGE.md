# Docker 使用指南

本文档说明如何在Docker容器中运行XGBoost2pre.py脚本。

## 📋 前置要求

1. **安装Docker和Docker Compose**
   - Docker Desktop (Windows/Mac) 或 Docker Engine (Linux)
   - Docker Compose (通常包含在Docker Desktop中)

2. **数据文件准备**
   - 训练数据：`dataset_广东省_0.01_part*.csv` 文件应放在项目根目录
   - 验证数据：`validate_ds/` 或 `validate_ds_fcday*/` 目录应放在项目根目录

## 🚀 快速开始

### 1. 构建Docker镜像

```bash
# 在项目根目录（week20/e_predict/）执行
docker-compose build
```

或者直接使用docker命令：

```bash
docker build -t xgboost-predict .
```

### 2. 运行训练任务

#### 方式一：使用docker-compose（推荐）

```bash
# 基础训练（使用默认参数）
docker-compose run --rm xgboost-train python XGBoost2pre.py --normalize

# 使用PCA降维
docker-compose run --rm xgboost-train python XGBoost2pre.py --use_pca --n_components 10 --normalize

# 使用t-SNE降维
docker-compose run --rm xgboost-train python XGBoost2pre.py --use_tsne --n_components 10 --normalize

# 自定义学习率和正则化参数
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --lr 0.3 \
  --reg_alpha 0.1 \
  --reg_lambda 1.5 \
  --normalize

# 限制使用的part文件数量（用于快速测试）
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --max_parts 2 \
  --normalize

# 只进行预测（使用已训练的模型）
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --pred_only \
  --fcday 2 \
  --normalize

# 分析特征相关性
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --analyze_corr_only \
  --normalize

# 分析特征重要性
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --analyze_fea_imp \
  --normalize
```

#### 方式二：使用docker命令

```bash
# 构建镜像
docker build -t xgboost-predict .

# 运行容器
docker run --rm \
  -v "$(pwd):/app" \
  -e DOCKER_ENV=1 \
  xgboost-predict \
  python XGBoost2pre.py --normalize
```

### 3. 查看帮助信息

```bash
docker-compose run --rm xgboost-train python XGBoost2pre.py --help
```

## 📁 目录结构

在Docker容器中，项目结构如下：

```
/app/
├── XGBoost2pre.py          # 主脚本
├── requirements.txt         # Python依赖
├── dataset_广东省_0.01_part*.csv  # 训练数据（需要放在这里）
├── validate_ds/            # 验证数据目录
├── validate_ds_fcday2/     # 第2天预测验证数据
├── validate_ds_fcday3/     # 第3天预测验证数据
└── output/                 # 输出目录（自动创建）
    ├── saved_models/       # 保存的模型
    ├── preprocessors/       # 预处理器（scaler等）
    ├── feature_correlations/  # 特征相关性图
    ├── feature_importance/   # 特征重要性
    ├── learning_curves/      # 学习曲线图
    ├── train_metrics/        # 训练指标
    ├── test_metrics/          # 测试指标
    └── test_results/         # 测试结果
```

## 🔧 常用参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--normalize` | 启用数据归一化 | `--normalize` |
| `--use_pca` | 使用PCA降维 | `--use_pca --n_components 10` |
| `--use_tsne` | 使用t-SNE降维 | `--use_tsne --n_components 10` |
| `--lr` | 学习率 | `--lr 0.3` |
| `--reg_alpha` | L1正则化系数 | `--reg_alpha 0.1` |
| `--reg_lambda` | L2正则化系数 | `--reg_lambda 1.5` |
| `--max_parts` | 限制使用的part文件数量 | `--max_parts 2` |
| `--pred_only` | 只进行预测（不训练） | `--pred_only` |
| `--fcday` | 使用第几天的预测辐射 | `--fcday 2` |
| `--analyze_corr_only` | 只分析特征相关性 | `--analyze_corr_only` |
| `--analyze_fea_imp` | 分析特征重要性 | `--analyze_fea_imp` |
| `--feature_list` | 指定使用的特征（逗号分隔） | `--feature_list "solar_radiation,hour,month"` |
| `--enable_advanced_features` | 启用高级特征工程 | `--enable_advanced_features` |

## 📊 输出文件位置

所有输出文件都会保存在 `./output/` 目录下，包括：

- **模型文件**: `output/saved_models/`
- **预处理器**: `output/preprocessors/`
- **特征相关性图**: `output/feature_correlations/`
- **特征重要性**: `output/feature_importance/`
- **学习曲线**: `output/learning_curves/`
- **训练指标**: `output/train_metrics/`
- **测试指标**: `output/test_metrics/`
- **测试结果**: `output/test_results/`

## 🐛 故障排除

### 1. 找不到数据文件

**问题**: 脚本提示找不到数据文件

**解决方案**:
- 确保数据文件在项目根目录下
- 检查文件名格式是否正确：`dataset_广东省_0.01_part*.csv`
- 检查Docker volumes挂载是否正确

### 2. 权限问题

**问题**: 无法写入输出目录

**解决方案**:
```bash
# 确保输出目录有写权限
chmod -R 777 output/
```

### 3. 内存不足

**问题**: 容器运行过程中内存不足

**解决方案**:
- 使用 `--max_parts` 参数限制数据量
- 增加Docker的内存限制（Docker Desktop设置中）

### 4. 中文字体显示问题

**问题**: 图表中的中文显示为方块

**解决方案**:
- Dockerfile已安装中文字体，如果仍有问题，检查matplotlib配置

## 💡 最佳实践

1. **数据准备**: 确保所有数据文件都在项目根目录下
2. **参数调优**: 先用小数据集（`--max_parts 1`）测试，确认无误后再使用全部数据
3. **资源监控**: 训练大模型时监控CPU和内存使用情况
4. **结果备份**: 定期备份 `output/` 目录中的重要结果

## 🔄 更新镜像

如果修改了代码或依赖，需要重新构建镜像：

```bash
docker-compose build --no-cache
```

## 📝 示例：完整训练流程

```bash
# 1. 构建镜像
docker-compose build

# 2. 使用小数据集快速测试
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --max_parts 1 \
  --normalize \
  --analyze_fea_imp

# 3. 正式训练（使用全部数据）
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --normalize \
  --lr 0.3 \
  --reg_alpha 0 \
  --reg_lambda 1

# 4. 在测试集上评估
docker-compose run --rm xgboost-train python XGBoost2pre.py \
  --pred_only \
  --fcday 2 \
  --normalize
```

## 🆘 获取帮助

如果遇到问题，可以：

1. 查看脚本帮助：`docker-compose run --rm xgboost-train python XGBoost2pre.py --help`
2. 检查容器日志：`docker-compose logs xgboost-train`
3. 进入容器调试：`docker-compose run --rm xgboost-train /bin/bash`

