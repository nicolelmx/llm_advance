"""
为 Project_B 创建低风险和高风险样本
使用 Project_A 的模型和数据
"""
import json
import pathlib
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# 路径配置
PROJECT_B_ROOT = pathlib.Path(__file__).resolve().parent
PROJECT_A_ROOT = PROJECT_B_ROOT.parent / "Project_A"
MODEL_DIR = PROJECT_A_ROOT / "models"
DATA_DIR = PROJECT_A_ROOT / "data" / "processed"
ORIGINAL_DATA = PROJECT_A_ROOT / "data" / "creditcard.csv"
OUTPUT_DIR = PROJECT_B_ROOT / "sample_data"
OUTPUT_DIR.mkdir(exist_ok=True)


def load_model_and_scaler():
    """加载模型和标准化器"""
    model_path = MODEL_DIR / "lgbm_model.pkl"
    scaler_path = DATA_DIR / "scaler.joblib"
    columns_path = DATA_DIR / "columns.json"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(columns_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names


def predict_sample(model, scaler, feature_names, sample_data):
    """预测单个样本"""
    df = pd.DataFrame([sample_data])
    df = df[feature_names]
    
    # 标准化
    X_scaled = scaler.transform(df.values)
    
    # 预测
    try:
        proba = model.predict_proba(X_scaled)
        score = float(proba[0][1])  # 欺诈概率
    except AttributeError:
        proba = model.predict(X_scaled)
        score = float(proba[0])
    
    label = int(score >= 0.5)
    return score, label


def find_samples(model, scaler, feature_names, X_test, y_test):
    """找到低风险和高风险样本"""
    # 预测所有测试样本
    X_scaled = scaler.transform(X_test[feature_names].values)
    
    try:
        proba = model.predict_proba(X_scaled)
        scores = proba[:, 1]  # 欺诈概率
    except AttributeError:
        proba = model.predict(X_scaled)
        scores = proba
    
    # 找到高风险样本（score > 0.7 且真实标签为1的欺诈样本）
    high_risk_mask = (scores > 0.7) & (y_test.values == 1)
    high_risk_indices = y_test[high_risk_mask].index
    
    if len(high_risk_indices) > 0:
        high_risk_idx = high_risk_indices[0]
        high_risk_sample = X_test.loc[high_risk_idx]
        high_risk_score = scores[high_risk_mask][0]
        high_risk_label = y_test.loc[high_risk_idx]
        print(f"✓ 找到高风险样本，索引: {high_risk_idx}, Score: {high_risk_score:.6f}")
    else:
        # 如果没有找到，使用 score 最高的欺诈样本
        fraud_mask = y_test.values == 1
        if fraud_mask.sum() > 0:
            fraud_scores = scores[fraud_mask]
            fraud_indices = y_test[fraud_mask].index
            max_idx = fraud_scores.argmax()
            high_risk_idx = fraud_indices[max_idx]
            high_risk_sample = X_test.loc[high_risk_idx]
            high_risk_score = float(fraud_scores[max_idx])
            high_risk_label = y_test.loc[high_risk_idx]
            print(f"✓ 找到高风险样本（使用最高score的欺诈样本），索引: {high_risk_idx}, Score: {high_risk_score:.6f}")
        else:
            print("✗ 未找到欺诈样本")
            return None, None
    
    # 找到低风险样本（score < 0.1 且真实标签为0的正常样本）
    low_risk_mask = (scores < 0.1) & (y_test.values == 0)
    low_risk_indices = y_test[low_risk_mask].index
    
    if len(low_risk_indices) > 0:
        low_risk_idx = low_risk_indices[0]
        low_risk_sample = X_test.loc[low_risk_idx]
        low_risk_score = scores[low_risk_mask][0]
        low_risk_label = y_test.loc[low_risk_idx]
        print(f"✓ 找到低风险样本，索引: {low_risk_idx}, Score: {low_risk_score:.6f}")
    else:
        # 如果没有找到，使用 score 最低的正常样本
        normal_mask = y_test.values == 0
        if normal_mask.sum() > 0:
            normal_scores = scores[normal_mask]
            normal_indices = y_test[normal_mask].index
            min_idx = normal_scores.argmin()
            low_risk_idx = normal_indices[min_idx]
            low_risk_sample = X_test.loc[low_risk_idx]
            low_risk_score = float(normal_scores[min_idx])
            low_risk_label = y_test.loc[low_risk_idx]
            print(f"✓ 找到低风险样本（使用最低score的正常样本），索引: {low_risk_idx}, Score: {low_risk_score:.6f}")
        else:
            print("✗ 未找到正常样本")
            return None, None
    
    return {
        'high_risk': {
            'sample': high_risk_sample,
            'score': high_risk_score,
            'label': int(high_risk_label),
            'true_label': int(high_risk_label)
        },
        'low_risk': {
            'sample': low_risk_sample,
            'score': low_risk_score,
            'label': int(low_risk_label),
            'true_label': int(low_risk_label)
        }
    }


def main():
    print("=" * 70)
    print("为 Project_B 创建低风险和高风险样本")
    print("=" * 70)
    print()
    
    # 加载原始数据
    print("1. 加载原始数据...")
    df = pd.read_csv(ORIGINAL_DATA)
    y = df["Class"]
    X = df.drop(columns=["Class"])
    
    # 使用相同的参数分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"   测试集大小: {len(X_test)} 行")
    print()
    
    # 加载模型和 scaler
    print("2. 加载模型和标准化器...")
    model, scaler, feature_names = load_model_and_scaler()
    print(f"   特征数量: {len(feature_names)}")
    print()
    
    # 找到样本
    print("3. 查找低风险和高风险样本...")
    samples = find_samples(model, scaler, feature_names, X_test, y_test)
    
    if samples is None:
        print("\n✗ 无法找到合适的样本")
        return
    
    print()
    
    # 保存高风险样本
    print("4. 保存样本文件...")
    high_risk_sample = samples['high_risk']['sample']
    high_risk_features = {col: float(high_risk_sample[col]) for col in feature_names}
    
    high_risk_output = {
        "risk_level": "高风险",
        "true_label": samples['high_risk']['true_label'],
        "predicted_score": samples['high_risk']['score'],
        "predicted_label": samples['high_risk']['label'],
        "features": high_risk_features,
        "_note": "这是从原始数据中提取的真实高风险样本（欺诈样本），特征值为原始值（未标准化），API 会自动进行标准化"
    }
    
    high_risk_file = OUTPUT_DIR / "高风险_sample.json"
    with open(high_risk_file, "w", encoding="utf-8") as f:
        json.dump(high_risk_output, f, indent=2, ensure_ascii=False)
    print(f"   ✓ 已保存高风险样本: {high_risk_file.name}")
    print(f"      Score: {samples['high_risk']['score']:.6f}, Label: {samples['high_risk']['label']}")
    
    # 保存低风险样本
    low_risk_sample = samples['low_risk']['sample']
    low_risk_features = {col: float(low_risk_sample[col]) for col in feature_names}
    
    low_risk_output = {
        "risk_level": "低风险",
        "true_label": samples['low_risk']['true_label'],
        "predicted_score": samples['low_risk']['score'],
        "predicted_label": samples['low_risk']['label'],
        "features": low_risk_features,
        "_note": "这是从原始数据中提取的真实低风险样本（正常样本），特征值为原始值（未标准化），API 会自动进行标准化"
    }
    
    low_risk_file = OUTPUT_DIR / "低风险_sample.json"
    with open(low_risk_file, "w", encoding="utf-8") as f:
        json.dump(low_risk_output, f, indent=2, ensure_ascii=False)
    print(f"   ✓ 已保存低风险样本: {low_risk_file.name}")
    print(f"      Score: {samples['low_risk']['score']:.6f}, Label: {samples['low_risk']['label']}")
    
    print()
    print("=" * 70)
    print("完成！")
    print("=" * 70)
    print()
    print("使用说明：")
    print("1. 高风险样本文件: sample_data/高风险_sample.json")
    print("2. 低风险样本文件: sample_data/低风险_sample.json")
    print("3. 在 Project_B API 的 Swagger UI 中测试这些样本")
    print("4. 高风险样本应该返回 score > 0.7, label = 1")
    print("5. 低风险样本应该返回 score < 0.1, label = 0")
    print()
    print("API 测试地址: http://localhost:8001/docs")


if __name__ == "__main__":
    main()

