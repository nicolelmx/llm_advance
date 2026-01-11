from datetime import datetime, timedelta  # 用于处理时间差
import glob
import joblib  # 用于保存和加载机器学习模型相关处理器的
import os
import warnings
import traceback  # 获取错误部分的详细信息，便于debug调试用
from pathlib import Path

from sklearn.decomposition import PCA  # 主成分分析 降维操作
from sklearn.manifold import TSNE  # 做非线性降维和可视化
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 做归一化
from pandas.errors import SettingWithCopyWarning

# UMAP导入 - 暂时禁用UMAP功能
UMAP = None

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns  # 用于统计可视化
import xgboost as xgb

'''
数据足够，样本的量，时序数据的时间周期
情况一：设计一个模型，可以预测未来一天的发电情况
简单：机器学习的模型 XGBoost最好
'''


# 🆕 Docker支持：获取基础路径（支持本地和Docker环境）
def get_base_path():
    """获取项目基础路径，支持本地和Docker

    r环境"""
    # 检查是否在Docker环境中（通过环境变量或路径判断）
    if os.getenv('DOCKER_ENV') == '1' or os.path.exists('/app'):
        base_path = Path('/app')
    else:
        # 本地环境：使用脚本所在目录
        base_path = Path(__file__).resolve().parent
    return base_path


BASE_PATH = get_base_path()

# 设置中文显示
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 忽略SettingWithCopyWarning警告
warnings.filterwarnings(action='ignore', category=SettingWithCopyWarning)

# ===================== Docker环境路径配置 =====================
# 统一使用BASE_PATH，兼容字符串和Path对象
BASE_DIR = str(BASE_PATH)  # 为了兼容使用os.path.join的旧代码


# 数据预处理
def preprocess_data(df):
    # 处理时间相关特征
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 编码分类特征
    label_encoders = {}
    categorical_cols = [
        "solar_intensity",
        # 'Province', 'City', 'Region'
    ]
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = df[col].fillna("missing")
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 布尔特征转数值
    df["is_daytime"] = df["is_daytime"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    # 填充气象特征缺失值
    weather_cols = [
        "solar_radiation",
        "solar_mean_1h",
        "solar_max_1h",
        "solar_min_1h",
        "solar_std_1h",
        "solar_mean_3h",
        "solar_max_3h",
        "solar_min_3h",
        "solar_std_3h",
        "solar_mean_6h",
        "solar_max_6h",
        "solar_min_6h",
        "solar_std_6h",
        "solar_diff",
    ]

    for col in weather_cols:
        df[col] = df.groupby(["station_id", "hour"])[col].transform(
            lambda x: x.fillna(x.mean())
        )
        df[col] = df[col].fillna(0)

    # 处理特殊列
    if "solar_intensity" in df.columns:
        df["solar_intensity"] = df["solar_intensity"].fillna("missing")

    return df


def create_advanced_features(df):
    """创建高级时间序列特征 - 这是性能提升的关键！"""
    print("--> 开始创建高级时间序列特征...")
    df = df.sort_index()  # 确保按时间排序

    # 1. 滞后特征 (Lag Features) - 时间序列预测的核心
    print("  --> 创建滞后特征...")
    for lag in [1, 2, 4, 24, 96]:  # 15分钟, 30分钟, 1小时, 6小时, 24小时
        df[f'power_lag_{lag}'] = df.groupby('station_id')['power_output'].shift(lag)
        df[f'radiation_lag_{lag}'] = df.groupby('station_id')['solar_radiation'].shift(lag)

    # 2. 滑动窗口统计特征 (Rolling Window Features)
    print("  --> 创建滑动窗口特征...")
    for window in [4, 12, 24, 96]:  # 1小时, 3小时, 6小时, 24小时
        # 辐射相关的滑动窗口
        df[f'radiation_roll_mean_{window}'] = df.groupby('station_id')['solar_radiation'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'radiation_roll_max_{window}'] = df.groupby('station_id')['solar_radiation'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
        df[f'radiation_roll_std_{window}'] = df.groupby('station_id')['solar_radiation'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )

        # 功率相关的滑动窗口
        # df[f'power_roll_mean_{window}'] = df.groupby('station_id')['power_output'].transform(
        #     lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        # ) # 数据泄露风险：移除该特征

    # 3. 交互特征 (Interaction Features)
    print("  --> 创建交互特征...")
    df['hour_x_radiation'] = df['hour'] * df['solar_radiation']
    df['season_x_radiation'] = df['season'] * df['solar_radiation']
    # df['radiation_efficiency'] = df['power_output'] / (df['solar_radiation'] + 0.001)  # 严重的数据泄露，已移除

    # 4. 更精细的时间特征
    print("  --> 创建精细时间特征...")
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int)
    df['quarter'] = df.index.quarter
    df['day_of_month'] = df.index.day

    # 5. 辐射变化特征
    print("  --> 创建辐射变化特征...")
    df['radiation_diff_1h'] = df.groupby('station_id')['solar_radiation'].diff(4)  # 1小时变化
    df['radiation_diff_ratio'] = df['radiation_diff_1h'] / (df['solar_radiation'] + 0.001)

    # 6. 季节性特征
    print("  --> 创建季节性特征...")
    df['is_summer'] = ((df.index.month >= 6) & (df.index.month <= 8)).astype(int)
    df['is_winter'] = ((df.index.month >= 12) | (df.index.month <= 2)).astype(int)

    # 用合理的值填充NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print(
        f"--> 特征工程完成！新增 {len([col for col in df.columns if any(keyword in col for keyword in ['lag_', 'roll_', '_x_', 'efficiency', 'dayofyear', 'weekofyear', 'diff_', 'is_'])])} 个高级特征")
    return df


# 新增：计算并保存特征相关性
def analyze_feature_correlation(X, features, province_name, station_id):
    # 计算相关系数矩阵
    corr_matrix = pd.DataFrame(X, columns=features).corr()  # 用于计算各个列之间的皮尔逊相关系数

    # 创建输出目录 - 使用相对路径（支持Docker）
    output_dir = os.path.join(BASE_DIR, "output", "feature_correlations")
    os.makedirs(output_dir, exist_ok=True)

    # 绘制并保存热力图
    plt.figure(figsize=(20, 15))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title(f'{province_name} - 站点{station_id} 特征相关性矩阵')
    plot_path = os.path.join(output_dir, f"{province_name}_station{station_id}_correlation.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"特征相关性图已保存至: {plot_path}")

    # 找出相关性>0.9的特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))

    # 打印高相关特征对
    if high_corr_pairs:
        print("\n高相关性特征对(相关系数>0.9):")
        for pair in high_corr_pairs:
            print(f"{pair[0]} 和 {pair[1]}: {pair[2]:.4f}")
    else:
        print("\n没有发现相关系数>0.9的特征对")


# 定义特征和目标
def get_features_target(
        df,
        use_pca=False,
        use_tsne=False,
        use_umap=False,
        n_components=None,
        variance_threshold=0.95,
        dim_reducer=None,
        y_scaler=None,
        X_scaler=None,
        province_name=None,
        station_id=None,
        analyze_corr_only=False,
        normalize=True,
        feature_list=None  # 🆕 新增：动态特征选择
):
    # 🆕 智能特征选择逻辑
    if feature_list is None:
        # 默认特征集合（基础特征）
        base_features = [
            "solar_radiation", "hour", "hour_sin", "hour_cos",
            "month", "season", "solar_mean_1h", "solar_max_1h", "solar_min_1h",
            "solar_mean_3h", "solar_max_3h", "solar_min_3h",
            "solar_mean_6h", "solar_max_6h", "solar_min_6h", "solar_std_6h",
            "solar_diff", "solar_intensity"
        ]

        # 高级特征集合（如果存在的话）
        advanced_features = [
            'radiation_lag_1', 'radiation_lag_2', 'radiation_lag_4', 'radiation_lag_24', 'radiation_lag_96',
            # 滑动窗口特征
            'radiation_roll_mean_4', 'radiation_roll_mean_12', 'radiation_roll_mean_24', 'radiation_roll_mean_96',
            'radiation_roll_max_4', 'radiation_roll_max_12', 'radiation_roll_max_24', 'radiation_roll_max_96',
            'radiation_roll_std_4', 'radiation_roll_std_12', 'radiation_roll_std_24', 'radiation_roll_std_96',
            # 'power_roll_mean_4', 'power_roll_mean_12', 'power_roll_mean_24', 'power_roll_mean_96', # Leakage Risk
            # 交互特征
            'hour_x_radiation', 'season_x_radiation',
            # 'radiation_efficiency', # Severe Leakage Risk
            # 时间特征
            'dayofyear', 'weekofyear', 'quarter', 'day_of_month',
            # 变化特征
            'radiation_diff_1h', 'radiation_diff_ratio',
            # 季节特征
            'is_summer', 'is_winter'
        ]

        # 组合所有可用特征
        all_possible_features = base_features + advanced_features
        features = [f for f in all_possible_features if f in df.columns]

        print(f"--> 使用默认特征集：{len(features)} 个特征")
    else:
        # 使用Agent指定的特征列表
        features = [f for f in feature_list if f in df.columns]
        missing_features = [f for f in feature_list if f not in df.columns]

        print(f"--> 使用Agent指定特征集：{len(features)} 个特征")
        if missing_features:
            print(f"⚠️  缺失特征：{missing_features}")

    print(f"--> 最终特征列表：{features[:10]}{'...' if len(features) > 10 else ''}")  # 只显示前10个
    X = df[features]
    y = df["power_output"] * 6

    # 在训练模式下执行相关性分析
    if X_scaler is None and not use_pca and not use_tsne and analyze_corr_only:
        analyze_feature_correlation(X, features, province_name, station_id)

    # 添加特征标准化 (除时间周期特征外)
    non_cyclic_features = [f for f in features if f not in ["hour_sin", "hour_cos"]]
    if normalize and X_scaler is None:  # 训练模式
        X_scaler = StandardScaler()
        X[non_cyclic_features] = X_scaler.fit_transform(X[non_cyclic_features])
    elif normalize and X_scaler is not None:  # 测试模式
        X[non_cyclic_features] = X_scaler.transform(X[non_cyclic_features])

    # 标签归一化
    if normalize and y_scaler is None:
        y_scaler = StandardScaler()
        y = y_scaler.fit_transform(y.values.reshape(-1, 1)).flatten()
    elif normalize and y_scaler is not None:
        y = y_scaler.transform(y.values.reshape(-1, 1)).flatten()

    # 特征降维
    if use_pca:
        if dim_reducer is None:  # 训练模式
            if variance_threshold is not None:
                # 计算PCA的最佳维度
                pca = PCA()
                pca.fit(X)

                # 计算累计方差贡献率
                explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.argmax(explained_variance_ratio >= variance_threshold) + 1

                print(
                    f"PCA降维: 保留{variance_threshold * 100}%方差需要{n_components}个主成分"
                )
                # print(f"各主成分方差贡献率: {pca.explained_variance_ratio_}")

            # 重新用最佳维度拟合
            dim_reducer = PCA(n_components=n_components)
            X = dim_reducer.fit_transform(X)
            print(f"PCA降维后特征数: {X.shape[1]}")
        else:  # 测试模式
            X = dim_reducer.transform(X)
    elif use_tsne:
        print("start tsne, ", datetime.now())
        # if dim_reducer is None:  # 训练模式
        if n_components >= 4:
            dim_reducer = TSNE(n_components=n_components, method='exact',
                               #    random_state=42
                               )
        else:
            dim_reducer = TSNE(n_components=n_components,
                               #    random_state=42
                               )

        X = dim_reducer.fit_transform(X)
        print("end tsne, ", datetime.now())
        # else:  # 测试模式
        #     X = dim_reducer.transform(X)  # ERROR! TODO 不学习一个固定的映射函数（不像 PCA 或 LDA），因此无法直接对新数据进行转换
        print(f"t-SNE降维后特征数: {X.shape[1]}")
    elif use_umap:  # 新增UMAP降维
        if UMAP is None:
            raise ImportError("UMAP库未安装，无法使用UMAP降维")
        if dim_reducer is None:  # 训练模式
            dim_reducer = UMAP(n_components=n_components)
            X = dim_reducer.fit_transform(X)
        else:  # 测试模式
            X = dim_reducer.transform(X)
        print(f"UMAP降维后特征数: {X.shape[1]}")

    return X, y, features, y_scaler, dim_reducer, X_scaler


def calculate_metrics(test_df, y_true, y_pred, plant_scales):
    # 确保输入为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    plant_scales = np.array(plant_scales)

    # 保留原始功率相关指标
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # 计算容量利用率相关指标
    capacity_true = y_true / plant_scales
    capacity_pred = y_pred / plant_scales
    capacity_mae = mean_absolute_error(capacity_true, capacity_pred)
    capacity_rmse = np.sqrt(mean_squared_error(capacity_true, capacity_pred))

    # 计算图示公式3中的误差指标
    timestamps = test_df.index  # 假设索引是时间戳
    station_ids = test_df["station_id"].values
    formula3_score = calculate_daily_formula3_score(
        y_true, y_pred, plant_scales, timestamps, station_ids
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Capacity_MAE": float(capacity_mae),
        "Capacity_RMSE": float(capacity_rmse),
        "Formula3_Score": float(formula3_score),
    }


def calculate_daily_formula3_score(
        y_true, y_pred, plant_scales, timestamps, station_ids
):
    """
    计算每个电站每天的公式3评分，然后返回所有日期的平均分

    参数:
    y_true: 实际功率输出数组
    y_pred: 预测功率输出数组
    plant_scales: 每个数据点对应的电站装机容量数组
    timestamps: 每个数据点对应的时间戳数组（需与y_true等长）
    station_ids: 每个数据点对应的电站ID数组

    返回:
    所有电站日评分的平均值
    """
    # 创建包含所有数据的DataFrame
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "station_id": station_ids,
            "y_true": y_true,
            "y_pred": y_pred,
            "plant_scale": plant_scales,
        }
    )

    # 转换时间戳并提取日期
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date

    # 筛选6:00-19:00之间的数据点（每15分钟一个）
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df = df[(df["hour"] >= 6) & (df["hour"] <= 19)]

    # 按电站和日期分组计算每日评分
    daily_scores = []
    for (station_id, date), group in df.groupby(["station_id", "date"]):
        if len(group) < 40:
            continue

        group_errors = []
        for _, row in group.iterrows():
            cap = row["plant_scale"]
            threshold = 0.2 * cap

            if row["y_true"] >= threshold:
                error = (row["y_pred"] - row["y_true"]) / row["y_true"]
            else:
                error = (row["y_pred"] - row["y_true"]) / (0.2 * cap)

            group_errors.append(error ** 2)

        # 计算单日评分
        rmse = np.sqrt(np.mean(group_errors))
        daily_score = 1 - rmse
        daily_scores.append(daily_score)

    # 返回所有日期的平均评分
    return np.mean(daily_scores) if daily_scores else 0


def save_preprocessors(province_name, station_id, y_scaler, X_scaler, dim_reducer, use_pca, use_tsne, use_umap,
                       n_components):
    """保存预处理实例到磁盘"""
    # 使用相对路径（支持Docker）
    preprocessors_dir = BASE_PATH / "output" / "preprocessors" / province_name
    os.makedirs(preprocessors_dir, exist_ok=True)

    # 保存归一化实例
    joblib.dump(y_scaler, str(preprocessors_dir / f"y_scaler_{station_id}.joblib"))
    joblib.dump(X_scaler, str(preprocessors_dir / f"X_scaler_{station_id}.joblib"))

    # 保存降维实例
    if use_pca:
        joblib.dump(dim_reducer, str(preprocessors_dir / f"pca_{station_id}.joblib"))
    elif use_tsne:
        joblib.dump(dim_reducer, str(preprocessors_dir / f"tsne_{station_id}.joblib"))
    elif use_umap:  # 新增UMAP保存
        joblib.dump(dim_reducer, str(preprocessors_dir / f"umap_{station_id}.joblib"))


def load_preprocessors(province_name, station_id, use_pca, use_tsne, use_umap):
    """从磁盘加载预处理实例"""
    # 使用相对路径（支持Docker）
    preprocessors_dir = BASE_PATH / "output" / "preprocessors" / province_name

    try:
        y_scaler = joblib.load(str(preprocessors_dir / f"y_scaler_{station_id}.joblib"))
        X_scaler = joblib.load(str(preprocessors_dir / f"X_scaler_{station_id}.joblib"))

        if use_pca:
            dim_reducer = joblib.load(str(preprocessors_dir / f"pca_{station_id}.joblib"))
        elif use_tsne:
            dim_reducer = joblib.load(str(preprocessors_dir / f"tsne_{station_id}.joblib"))
        elif use_umap:  # 新增UMAP加载
            dim_reducer = joblib.load(str(preprocessors_dir / f"umap_{station_id}.joblib"))
        else:
            dim_reducer = None

        return y_scaler, X_scaler, dim_reducer
    except FileNotFoundError:
        return None, None, None


# 处理每个省份数据
def process_province_data(
        province_name="广东省",
        use_pca=False,
        use_tsne=False,
        use_umap=False,
        lr=0.3,
        reg_alpha=0,
        reg_lambda=1,
        max_parts=None,
        sample_rate=0.01,
        variance_threshold=0.95,
        n_components=None,
        analyze_corr_only=False,
        analyze_fea_imp=False,
        normalize=True,
        feature_list=None,  # 🆕 新增：动态特征选择
        enable_advanced_features=True  # 🆕 新增：是否启用高级特征工程
):
    print(f"\n{'=' * 50}")
    print(f"开始处理省份: {province_name}")
    start_time = datetime.now()
    print(f"开始时间: {start_time}")

    # 查找广东省的所有part文件 - 使用相对路径（支持Docker）
    data_pattern = str(BASE_PATH / "dataset_广东省_0.01_part*.csv")
    print(f"查找数据文件模式: {data_pattern}")

    # 使用glob查找匹配的文件
    data_files = sorted(glob.glob(data_pattern))  # 确保文件按顺序排列
    print(f"找到 {len(data_files)} 个数据文件")

    if not data_files:
        print(f"未找到 {province_name} 的数据文件，跳过")
        print("请检查:")
        print(f"1. 当前工作目录: {os.getcwd()}")
        print(f"2. 基础路径: {BASE_PATH}")
        print(f"3. 数据文件是否在正确位置: {data_pattern}")
        return None

    # 显示找到的文件
    print("前5个文件:")
    for i, file_path in enumerate(data_files[:5]):
        print(f"  {i + 1}. {os.path.basename(file_path)}")

    # 检查文件是否存在
    valid_files = []
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"+ 文件存在: {os.path.basename(file_path)}")
            valid_files.append(file_path)
        else:
            print(f"- 文件不存在: {os.path.basename(file_path)}")

    if not valid_files:
        print("错误: 没有找到有效的文件")
        return None

    data_files = valid_files

    # 限制使用的part文件数量
    if max_parts is not None:
        data_files = data_files[:max_parts]
        print(f"限制使用前 {max_parts} 个part文件")

    print(f"最终使用 {len(data_files)} 个数据文件")

    # 读取并合并所有part文件
    df_list = []
    total_records = 0

    for i, file_path in enumerate(data_files):
        try:
            print(f"正在读取文件 {i + 1}/{len(data_files)}: {os.path.basename(file_path)}")

            # 检查文件是否存在
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在: {file_path}")
                continue

            # 检查文件大小
            file_size = os.path.getsize(file_path)
            print(f"文件大小: {file_size / (1024 * 1024):.2f} MB")

            part_df = pd.read_csv(file_path, index_col=0)
            part_df.index = pd.to_datetime(part_df.index)

            print(f"文件列数: {len(part_df.columns)}, 行数: {len(part_df)}")
            print(f"列名: {list(part_df.columns)}")

            df_list.append(part_df)
            total_records += len(part_df)
            print(f"已加载文件: {os.path.basename(file_path)} ({len(part_df)} 条记录)")

        except Exception as e:
            print(f"读取文件 {file_path} 失败: {str(e)}")
            print(f"错误详情: {traceback.format_exc()}")

    if not df_list:
        print(f"错误: 无法加载 {province_name} 的任何数据文件")
        return None

    # 合并所有数据
    print(f"开始合并 {len(df_list)} 个数据文件...")
    df = pd.concat(df_list, ignore_index=False)
    print(f"合并后的数据集大小: {len(df)} 条记录")
    print(f"数据集列数: {len(df.columns)}")
    print(f"数据集时间范围: {df.index.min()} 到 {df.index.max()}")
    print(f"数据集列名: {list(df.columns)}")

    # 数据预处理
    print("开始数据预处理...")
    df_proc = preprocess_data(df)

    # 🆕 高级特征工程
    if enable_advanced_features:
        print("开始高级特征工程...")
        df_eng = create_advanced_features(df_proc)
    else:
        print("跳过高级特征工程，使用基础特征...")
        df_eng = df_proc.sort_index()

    print(f"特征工程后数据集大小: {len(df_eng)} 条记录")
    print(f"特征工程后特征数量: {len(df_eng.columns)} 个特征")

    # 检查是否有power_output列
    if "power_output" not in df_eng.columns:
        print("错误: 数据集中没有 'power_output' 列")
        print(f"可用列: {list(df_eng.columns)}")
        return None

    # 检查是否有station_id列
    if "station_id" not in df_eng.columns:
        print("错误: 数据集中没有 'station_id' 列")
        print(f"可用列: {list(df_eng.columns)}")
        return None

    station_results = []
    formula3_score_list = []
    # 初始化存储所有站点损失的列表
    all_train_loss = []
    all_test_loss = []

    # 获取所有站点
    stations = df_eng["station_id"].unique()
    print(f"发现 {len(stations)} 个站点: {stations[:10]}...")  # 只显示前10个站点

    for i, (station_id, station_data) in enumerate(df_eng.groupby("station_id")):
        try:
            print(f"\n处理站点 {i + 1}/{len(stations)}: {station_id}")

            station_data = station_data.sort_index()

            test_start = pd.to_datetime("2024-12-01")

            test_mask = station_data.index >= test_start
            train_mask = station_data.index < test_start

            train_df = station_data[train_mask]
            test_df = station_data[test_mask]

            # 打印数据时间范围
            print(f"训练集时间范围: {train_df.index.min()} 到 {train_df.index.max()}")
            print(f"测试集时间范围: {test_df.index.min()} 到 {test_df.index.max()}")
            print(f"训练集大小: {len(train_df)}, 测试集大小: {len(test_df)}")

            # 检查数据是否足够
            if len(train_df) < 100:
                print(f"警告: 站点 {station_id} 训练数据不足 ({len(train_df)} 条记录)，跳过")
                continue

            if len(test_df) < 10:
                print(f"警告: 站点 {station_id} 测试数据不足 ({len(test_df)} 条记录)，跳过")
                continue

            # 准备特征和目标
            X_train, y_train, features, y_scaler, dim_reducer, X_scaler = get_features_target(
                train_df,
                use_pca=use_pca,
                use_tsne=use_tsne,
                use_umap=use_umap,
                n_components=n_components,
                variance_threshold=variance_threshold,
                province_name=province_name,
                station_id=station_id,
                analyze_corr_only=analyze_corr_only,
                normalize=normalize,
                feature_list=feature_list  # 🆕 传递特征列表
            )
            if analyze_corr_only:
                continue
            X_test, y_test, *_ = get_features_target(
                test_df,
                use_pca=use_pca,
                use_tsne=use_tsne,
                use_umap=use_umap,
                n_components=n_components,
                dim_reducer=dim_reducer,
                y_scaler=y_scaler,
                X_scaler=X_scaler,
                normalize=normalize,
                feature_list=feature_list  # 🆕 传递特征列表
            )

            # 保存预处理实例
            save_preprocessors(
                province_name=province_name,
                station_id=station_id,
                y_scaler=y_scaler,
                X_scaler=X_scaler,
                dim_reducer=dim_reducer,
                use_pca=use_pca,
                use_tsne=use_tsne,
                use_umap=use_umap,
                n_components=n_components
            )

            # 训练站点特定的模型
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=lr,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective="reg:squarederror",
                n_jobs=20,
                reg_alpha=reg_alpha,  # L1正则化
                reg_lambda=reg_lambda,  # L2正则化
                eval_metric=["rmse"]
            )

            print(f"开始训练模型... (训练集: {len(X_train)} 条记录)")

            # 添加评估集用于早停
            eval_set = [(X_train, y_train), (X_test, y_test)]
            model.fit(
                X_train, y_train, eval_set=eval_set, verbose=False
            )

            print("模型训练完成")

            if sum((use_pca, use_tsne, use_umap)) == 0 and analyze_fea_imp:
                # 分析特征重要性
                feature_importance = model.feature_importances_

                # 创建特征重要性DataFrame
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': feature_importance
                }).sort_values('importance', ascending=False)

                # 打印最重要的10个特征
                print("\n特征重要性排名(前10):")
                print(importance_df.head(10))

                # 保存特征重要性结果
                importance_dir = BASE_PATH / "output" / "feature_importance" / province_name
                os.makedirs(importance_dir, exist_ok=True)
                importance_path = importance_dir / f"station{station_id}_importance.csv"
                importance_df.to_csv(str(importance_path), index=False)
                print(f"特征重要性已保存至: {importance_path}")

                # 绘制特征重要性图
                plt.figure(figsize=(12, 8))
                sns.barplot(x='importance', y='feature', data=importance_df.head(20))
                plt.title(f'{province_name} - 站点{station_id} 特征重要性')
                plot_path = importance_dir / f"station{station_id}_importance.png"
                plt.savefig(str(plot_path), bbox_inches='tight')
                plt.close()
                print(f"特征重要性图已保存至: {plot_path}")

            # 预测测试集并确保非负
            y_pred = model.predict(X_test)
            if normalize:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            else:
                y_true = y_test
            y_pred = np.maximum(y_pred, 0)
            y_true = np.maximum(y_true, 0)

            # 计算公式3评分 - 严格按照图示公式
            plant_scale = test_df["PlantScale"].iloc[0]
            result = calculate_metrics(test_df, y_true, y_pred, plant_scale)

            # 收集结果
            station_results.append(
                {
                    "station_id": station_id,
                    "formula3_score": result["Formula3_Score"],
                    "plant_scale": plant_scale,
                    "train_size": len(X_train),
                    "test_size": len(X_test),
                }
            )
            formula3_score_list.append(result["Formula3_Score"])
            if result["Formula3_Score"] > 0:
                # 收集损失数据
                loss_results = model.evals_result()
                all_train_loss.append(loss_results["validation_0"]["rmse"])
                all_test_loss.append(loss_results["validation_1"]["rmse"])

            print(
                f"站点 {station_id} 评分: MAE={result['MAE']:.4f}, RMSE={result['RMSE']:.4f}, R2={result['R2']:.4f}, Formula3={result['Formula3_Score']:.4f}")

            # 模型保存
            model_name = (
                f"xgboost_{province_name}_station{station_id}_"
                f"{'pca' if use_pca else 'tsne' if use_tsne else 'umap' if use_umap else 'raw'}_"
                f"ncomp{n_components}_"
                f"alpha{reg_alpha}_lambda{reg_lambda}.joblib"
            )
            # 使用相对路径（支持Docker）
            model_dir = BASE_PATH / "output" / "saved_models"
            model_path = model_dir / model_name
            os.makedirs(model_dir, exist_ok=True)
            joblib.dump(model, str(model_path))
            print(f"模型已保存至: {model_path}")

        except Exception as e:
            print(f"处理站点 {station_id} 时出错: {traceback.format_exc()}")

    formula3_score_array = np.array(formula3_score_list)
    bg_zero_formula3_score_array = formula3_score_array[formula3_score_array > 0]

    avg_score = np.mean(bg_zero_formula3_score_array) if len(bg_zero_formula3_score_array) > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"{province_name} 训练结果汇总:")
    print(f"{'=' * 60}")
    print(f"总站点数: {len(formula3_score_array)}")
    print(f"有效站点数(Formula3>0): {len(bg_zero_formula3_score_array)}")
    print(f"无效站点数(Formula3<=0): {len(formula3_score_array[formula3_score_array <= 0])}")
    print(f"平均Formula3准确率: {avg_score:.4f}")
    print(f"最高Formula3准确率: {np.max(bg_zero_formula3_score_array):.4f}")
    print(f"最低Formula3准确率: {np.min(bg_zero_formula3_score_array):.4f}")
    print(f"{'=' * 60}")

    # ==========================================================
    # ================  【重要】为 Agent 添加的输出 ================
    print(f"FINAL_SCORE: {avg_score}")
    # ==========================================================

    if len(formula3_score_array[formula3_score_array <= 0]) > 0:
        print(f"无效站点的Formula3准确率: {formula3_score_array[formula3_score_array <= 0]}")

    # 保存训练阶段重要指标
    train_metrics = {
        "province_name": [province_name],
        "公式3准确率<0的站数": [len(formula3_score_array[formula3_score_array <= 0])],
        "公式3准确率<0的站准确率": [formula3_score_array[formula3_score_array <= 0]],
        "公式3准确率>0的平均准确率": [np.mean(bg_zero_formula3_score_array)],
        "公式3准确率>0的站数": [len(bg_zero_formula3_score_array)]
    }
    train_metrics_df = pd.DataFrame(train_metrics)
    # 使用相对路径（支持Docker），包含所有参数
    train_metrics_dir = BASE_PATH / "output" / "train_metrics"
    train_metrics_path = train_metrics_dir / f"{province_name}_train_metrics_lr{lr:.3f}_alpha{reg_alpha:.3f}_lambda{reg_lambda:.3f}.csv"
    os.makedirs(train_metrics_dir, exist_ok=True)
    train_metrics_df.to_csv(str(train_metrics_path), index=False)
    print(f"训练阶段重要指标已保存至: {train_metrics_path}")

    # 计算平均损失曲线
    if all_train_loss and all_test_loss:
        avg_train_loss = np.mean(all_train_loss, axis=0)
        avg_test_loss = np.mean(all_test_loss, axis=0)
        # 计算平均训练损失和平均测试损失的最小值
        min_train_loss = np.min(avg_train_loss)
        min_test_loss = np.min(avg_test_loss)

        plt.figure(figsize=(12, 6))
        plt.plot(avg_train_loss, label="平均训练损失")
        plt.plot(avg_test_loss, label="平均测试损失")

        # 绘制平均训练损失最小值的横虚线
        plt.axhline(y=min_train_loss, color='r', linestyle='--', label=f"训练损失最小值: {min_train_loss:.4f}")
        # 绘制平均测试损失最小值的横虚线
        plt.axhline(y=min_test_loss, color='g', linestyle='--', label=f"测试损失最小值: {min_test_loss:.4f}")

        plt.legend()
        plt.ylabel("RMSE")
        plt.xlabel("迭代次数")
        plt.title(f"{province_name} 所有站点平均损失曲线")

        # 使用相对路径（支持Docker）
        plot_dir = BASE_PATH / "output" / "learning_curves"
        plot_path = plot_dir / f"{province_name}_avg_learning_curve.png"
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(str(plot_path))
        plt.close()
        print(f"平均损失曲线已保存至: {plot_path}")

    end_time = datetime.now()
    print(f"结束时间: {end_time} 训练时间: {end_time - start_time}")
    return result


def test_on_final_dataset(province_name="广东省", use_pca=False, use_tsne=False,
                          use_umap=False, sample_rate=0.01, n_components=10,
                          reg_alpha=0, reg_lambda=1, fcday=2, normalize=True):
    """在最终测试集上评估模型性能"""
    print(f"\n{'=' * 50}")
    print(f"开始在最终测试集上评估省份: {province_name}")
    start_time = datetime.now()

    # 加载测试集数据 - 使用相对路径（支持Docker）
    if fcday == 2:
        output_dir = BASE_PATH / "validate_ds"
    else:
        output_dir = BASE_PATH / f"validate_ds_fcday{fcday}"
    validate_pattern = str(output_dir / f"dataset_fc_{province_name}_{sample_rate}_part*.csv")
    validate_files = sorted(glob.glob(validate_pattern))

    if not validate_files:
        print(f"未找到 {province_name} 的测试集文件")
        return None

    # 读取并合并测试集
    validate_df_list = []
    for file_path in validate_files:
        try:
            part_df = pd.read_csv(file_path, index_col=0)
            part_df.index = pd.to_datetime(part_df.index)
            validate_df_list.append(part_df)
            print(f"已加载测试集文件: {os.path.basename(file_path)} ({len(part_df)} 条记录)")
        except Exception as e:
            print(f"读取测试集文件 {file_path} 失败: {str(e)}")

    if not validate_df_list:
        print(f"错误: 无法加载 {province_name} 的任何测试集文件")
        return None

    validate_df = pd.concat(validate_df_list)
    validate_df = preprocess_data(validate_df)
    validate_df = validate_df.sort_index()

    # 评估结果存储
    validate_results = []
    formula3_scores = []

    # 按站点处理
    for station_id, station_data in validate_df.groupby("station_id"):
        try:
            print(f"\n处理测试集站点: {station_id}")

            # 加载预处理实例
            y_scaler, X_scaler, dim_reducer = load_preprocessors(
                province_name, station_id, use_pca, use_tsne, use_umap
            )

            if y_scaler is None:
                print(f"警告: 未找到站点 {station_id} 的预处理实例，跳过")
                continue

            # 加载模型
            model_name = (
                f"xgboost_{province_name}_station{station_id}_"
                f"{'pca' if use_pca else 'tsne' if use_tsne else 'raw'}_"
                f"ncomp{n_components}_"
                f"alpha{reg_alpha}_lambda{reg_lambda}.joblib"
            )
            # 使用相对路径（支持Docker）
            model_dir = BASE_PATH / "output" / "saved_models"
            model_path = model_dir / model_name

            if not model_path.exists():
                print(f"警告: 未找到站点 {station_id} 的模型文件，跳过")
                continue

            model = joblib.load(str(model_path))

            # 准备测试数据
            X_test, y_test, *_ = get_features_target(
                station_data,
                use_pca=use_pca,
                use_tsne=use_tsne,
                use_umap=use_umap,
                dim_reducer=dim_reducer,
                y_scaler=y_scaler,
                X_scaler=X_scaler,
                normalize=normalize
            )

            # 预测并反归一化
            y_pred = model.predict(X_test)
            if normalize:
                y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = y_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
            else:
                y_true = y_test
            y_pred = np.maximum(y_pred, 0) # 保证y_pred的值非负
            y_true = np.maximum(y_true, 0)

            '''
            1. y_pred = 5
            y_pred = np.maximum(y_pred, 0) = 5
            
            2. y_pred = -1
            y_pred = np.maximum(y_pred, 0) = 0
            '''

            # 计算指标
            plant_scale = station_data["PlantScale"].iloc[0]
            result = calculate_metrics(station_data, y_true, y_pred, plant_scale)

            validate_results.append({
                "station_id": station_id,
                "formula3_score": result["Formula3_Score"],
                "MAE": result["MAE"],
                "RMSE": result["RMSE"],
                "R2": result["R2"],
                "plant_scale": plant_scale,
                "sample_count": len(station_data)
            })

            if result["Formula3_Score"] > 0:
                formula3_scores.append(result["Formula3_Score"])

            print(f"站点 {station_id} 测试结果: {result}")

        except Exception as e:
            print(f"处理测试集站点 {station_id} 时出错: {str(e)}")
            continue

    # 汇总结果
    if validate_results:
        results_df = pd.DataFrame(validate_results)
        positive_scores = results_df[results_df["formula3_score"] > 0]

        print("\n最终测试集汇总结果:")
        print(f"{province_name} final important result: 总测试站数: {len(results_df)}")
        print(f"{province_name} final important result: 公式3准确率>0的站数: {len(positive_scores)}")
        print(
            f"{province_name} final important result: 公式3准确率>0的平均准确率: {np.mean(positive_scores['formula3_score']) if len(positive_scores) > 0 else 0:.4f}")
        print(f"{province_name} final important result: 公式3准确率<=0的站数: {len(results_df) - len(positive_scores)}")

        # 保存测试阶段重要指标
        test_metrics = {
            "province_name": [province_name],
            "总测试站数": [len(results_df)],
            "公式3准确率>0的站数": [len(positive_scores)],
            "公式3准确率>0的平均准确率": [
                np.mean(positive_scores['formula3_score']) if len(positive_scores) > 0 else 0],
            "公式3准确率<=0的站数": [len(results_df) - len(positive_scores)]
        }
        test_metrics_df = pd.DataFrame(test_metrics)
        # 使用相对路径（支持Docker）
        test_metrics_dir = BASE_PATH / "output" / "test_metrics"
        test_metrics_path = test_metrics_dir / f"{province_name}_test_metrics.csv"
        os.makedirs(test_metrics_dir, exist_ok=True)
        test_metrics_df.to_csv(str(test_metrics_path), index=False)
        print(f"测试阶段重要指标已保存至: {test_metrics_path}")

        # 保存详细结果
        # 使用相对路径（支持Docker）
        test_results_dir = BASE_PATH / "output" / "test_results"
        result_path = test_results_dir / f"{province_name}_final_test_results.csv"
        os.makedirs(test_results_dir, exist_ok=True)
        results_df.to_csv(str(result_path), index=False)
        print(f"测试结果已保存至: {result_path}")

    end_time = datetime.now()
    print(f"\n测试完成, 耗时: {end_time - start_time}")
    return validate_results if validate_results else None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guangdong Province PV Power Prediction Model Training")
    parser.add_argument("--use_pca", action="store_true", help="Use PCA dimensionality reduction")
    parser.add_argument("--use_tsne", action="store_true", help="Use t-SNE dimensionality reduction")
    parser.add_argument("--use_umap", action="store_true", help="Use UMAP dimensionality reduction")
    parser.add_argument("--analyze_corr_only", action="store_true", help="Only analyze correlation coefficients")
    parser.add_argument("--analyze_fea_imp", action="store_true", help="Analyze feature importance")
    parser.add_argument("--n_components", type=int, default=10,
                        help="Number of components after dimensionality reduction")
    parser.add_argument("--sample_rate", type=float, default=0.01,
                        help="Data sampling rate (0.01 means 1 percent of data)")
    parser.add_argument("--variance_threshold", type=float, default=0.95, help="PCA variance threshold, usually 0.95")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--reg_alpha", type=float, default=0, help="L1 regularization coefficient")
    parser.add_argument("--reg_lambda", type=float, default=1, help="L2 regularization coefficient")
    parser.add_argument("--max_parts", type=int, default=None, help="Maximum number of part files to use")
    parser.add_argument('--fcday', type=int, default=2, help='Which day forecast radiation to use')
    parser.add_argument('--pred_only', action='store_true', help='Only perform prediction')
    parser.add_argument('--normalize', action='store_true', help='Perform normalization')
    parser.add_argument('--feature_list', type=str, default=None, help='Comma-separated list of features to use')
    parser.add_argument('--enable_advanced_features', action='store_true', default=True,
                        help='Enable advanced feature engineering')

    args = parser.parse_args()
    print("处理省份: 广东省")
    print(f"使用第几天的预测辐射: {args.fcday}")
    print("component: ", repr(args.n_components))
    print("是否进行归一化: ", args.normalize)

    # 检查参数冲突
    if sum([args.use_pca, args.use_tsne, args.use_umap]) > 1:  # 修改冲突检查
        raise ValueError("不能同时使用多种降维方法")

    # 处理特征列表参数
    feature_list = None
    if args.feature_list:
        feature_list = [f.strip() for f in args.feature_list.split(',')]
        print(f"--> 使用指定特征: {len(feature_list)} 个特征")
        print(f"--> 特征列表: {feature_list[:5]}{'...' if len(feature_list) > 5 else ''}")

    # 处理广东省数据
    try:
        if not args.pred_only:
            result = process_province_data(
                province_name="广东省",
                use_pca=args.use_pca,
                use_tsne=args.use_tsne,
                use_umap=args.use_umap,
                lr=args.lr,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                max_parts=args.max_parts,
                sample_rate=args.sample_rate,
                variance_threshold=args.variance_threshold,
                n_components=args.n_components,
                analyze_corr_only=args.analyze_corr_only,
                analyze_fea_imp=args.analyze_fea_imp,
                normalize=args.normalize,
                feature_list=feature_list,  # 🆕 传递特征列表
                enable_advanced_features=args.enable_advanced_features  # 🆕 传递特征工程开关
            )
        else:
            test_results = test_on_final_dataset(
                province_name="广东省",
                use_pca=args.use_pca,
                use_tsne=args.use_tsne,
                use_umap=args.use_umap,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                n_components=args.n_components,
                fcday=args.fcday,
                normalize=args.normalize,
            )
    except Exception as e:
        print(f"处理广东省数据时出错: {str(e)}")
