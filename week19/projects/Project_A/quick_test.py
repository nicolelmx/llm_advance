"""
快速测试：直接使用原始数据中的高风险样本
"""
import json
import pathlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ORIGINAL_DATA = PROJECT_ROOT / "data" / "creditcard.csv"

# 加载原始数据
print("加载原始数据...")
df = pd.read_csv(ORIGINAL_DATA)
y = df["Class"]
X = df.drop(columns=["Class"])

# 使用相同的参数分割
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 加载模型和 scaler
print("加载模型和 scaler...")
model = joblib.load(MODEL_DIR / "lgbm_model.pkl")
scaler = joblib.load(DATA_DIR / "scaler.joblib")

with open(DATA_DIR / "columns.json", "r", encoding="utf-8") as f:
    feature_names = json.load(f)

# 找一个真实的欺诈样本（高风险）
print("\n查找欺诈样本...")
fraud_indices = y_test[y_test == 1].index
if len(fraud_indices) > 0:
    idx = fraud_indices[0]
    sample = X_test.loc[idx]
    true_label = y_test.loc[idx]
    
    print(f"找到欺诈样本，索引: {idx}")
    print(f"Time: {sample['Time']}")
    print(f"Amount: {sample['Amount']}")
    print(f"V1: {sample['V1']}")
    
    # 标准化
    X_scaled = scaler.transform([sample[feature_names].values])
    
    # 预测
    try:
        proba = model.predict_proba(X_scaled)
        score = float(proba[0][1])
    except AttributeError:
        proba = model.predict(X_scaled)
        score = float(proba[0])
    
    label = int(score >= 0.5)
    
    print(f"\n预测结果:")
    print(f"  Score: {score:.6f}")
    print(f"  Label: {label}")
    print(f"  真实标签: {true_label}")
    
    # 保存为 JSON
    features_dict = {col: float(sample[col]) for col in feature_names}
    output = {
        "risk_level": "真实高风险样本",
        "true_label": int(true_label),
        "predicted_score": score,
        "predicted_label": label,
        "features": features_dict
    }
    
    output_file = PROJECT_ROOT / "sample_data" / "真实高风险样本.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n已保存到: {output_file}")
    print("\n现在可以在 API 中测试这个文件，应该会返回高 score")
else:
    print("未找到欺诈样本")

