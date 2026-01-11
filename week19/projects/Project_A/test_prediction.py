"""
直接测试预测逻辑，找出问题所在
"""
import json
import pathlib
import pandas as pd
import joblib
import lightgbm as lgb
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SAMPLE_FILE = PROJECT_ROOT / "sample_data" / "高风险_sample_1.json"

# 加载模型和 scaler
print("加载模型和 scaler...")
model_path = MODEL_DIR / "lgbm_model.pkl"
scaler_path = DATA_DIR / "scaler.joblib"
columns_path = DATA_DIR / "columns.json"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

with open(columns_path, "r", encoding="utf-8") as f:
    feature_names = json.load(f)

print(f"模型类型: {type(model)}")
print(f"特征数量: {len(feature_names)}")
print()

# 加载样本数据
print("加载样本数据...")
with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
    sample_data = json.load(f)

features = sample_data["features"]
print(f"样本特征数量: {len(features)}")
print(f"Time: {features['Time']}")
print(f"Amount: {features['Amount']}")
print(f"V1: {features['V1']}")
print()

# 创建 DataFrame
df = pd.DataFrame([features])
print("原始 DataFrame:")
print(df[['Time', 'Amount', 'V1']].head())
print()

# 检查特征顺序
if feature_names:
    missing = set(feature_names) - set(df.columns)
    if missing:
        print(f"缺少特征: {missing}")
    else:
        print("✓ 所有特征都存在")
    df = df[feature_names]
    print(f"特征顺序已调整: {list(df.columns[:5])}...")
    print()

# 标准化
print("标准化特征...")
X_scaled = scaler.transform(df.values)
print(f"标准化后的形状: {X_scaled.shape}")
print(f"标准化后的前5个特征值: {X_scaled[0][:5]}")
print()

# 预测
print("进行预测...")
try:
    # 尝试 predict_proba
    proba = model.predict_proba(X_scaled)
    print(f"predict_proba 结果形状: {proba.shape}")
    print(f"predict_proba 结果: {proba[0]}")
    if proba.shape[1] > 1:
        score = float(proba[0][1])
    else:
        score = float(proba[0][0])
    print(f"Score (使用 predict_proba): {score}")
except AttributeError:
    # 如果是 Booster 对象
    proba = model.predict(X_scaled)
    print(f"predict 结果形状: {proba.shape}")
    print(f"predict 结果: {proba[0]}")
    score = float(proba[0])
    print(f"Score (使用 predict): {score}")

label = int(score >= 0.5)
print()
print("=" * 60)
print(f"最终结果:")
print(f"  Score: {score}")
print(f"  Label: {label}")
print("=" * 60)

