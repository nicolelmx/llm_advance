"""
修复 sample_data 中的 JSON 文件，将标准化后的值替换为原始值
这样 API 才能正确预测（API 期望输入原始值，然后自己进行标准化）
"""
import json
import pathlib
import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ORIGINAL_DATA = PROJECT_ROOT / "data" / "creditcard.csv"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
COLUMNS_PATH = DATA_DIR / "columns.json"


def load_original_data():
    """加载原始数据并按照训练时的相同方式分割"""
    df = pd.read_csv(ORIGINAL_DATA)
    y = df["Class"]
    X = df.drop(columns=["Class"])
    
    # 使用与训练时相同的参数进行分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 合并训练集和测试集（用于查找样本）
    all_df = pd.concat([X_train, X_test], ignore_index=True)
    all_y = pd.concat([y_train, y_test], ignore_index=True)
    all_df["Class"] = all_y.values
    
    return all_df


def find_matching_sample(original_df, scaled_features):
    """在原始数据中找到与标准化特征最匹配的样本"""
    # 加载 scaler 和 feature_names
    scaler = joblib.load(DATA_DIR / "scaler.joblib")
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    # 标准化原始数据
    X_original = original_df[feature_names].values
    X_scaled = scaler.transform(X_original)
    
    # 将标准化后的特征转换为 DataFrame
    scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
    
    # 将输入的标准化特征转换为 Series
    scaled_series = pd.Series(scaled_features)
    
    # 计算欧氏距离，找到最接近的样本
    distances = np.sqrt(((scaled_df - scaled_series) ** 2).sum(axis=1))
    closest_idx = distances.idxmin()
    
    return original_df.iloc[closest_idx], distances[closest_idx]


def fix_sample_file(json_path, original_df):
    """修复单个样本文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    scaled_features = data["features"]
    
    # 找到匹配的原始样本
    original_sample, distance = find_matching_sample(original_df, scaled_features)
    
    # 提取原始特征值
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    original_features = {col: float(original_sample[col]) for col in feature_names}
    
    # 更新 JSON 数据
    data["features"] = original_features
    data["_note"] = "特征值已更新为原始值（未标准化），API 会自动进行标准化"
    
    # 保存修复后的文件
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ 已修复: {json_path.name} (匹配距离: {distance:.6f})")
    return original_features


def main():
    print("=" * 70)
    print("修复 sample_data 中的 JSON 文件")
    print("=" * 70)
    print("\n说明：")
    print("- sample_data 中的 JSON 文件包含的是标准化后的特征值")
    print("- 但 API 期望输入原始值（未标准化），然后自己进行标准化")
    print("- 本脚本会将标准化后的值替换为原始值\n")
    
    # 加载原始数据
    print("加载原始数据...")
    original_df = load_original_data()
    print(f"原始数据大小: {len(original_df)} 行\n")
    
    # 查找所有 JSON 文件
    json_files = list(SAMPLE_DATA_DIR.glob("*.json"))
    
    if not json_files:
        print(f"[X] 在 {SAMPLE_DATA_DIR} 中未找到 JSON 文件")
        return
    
    print(f"找到 {len(json_files)} 个 JSON 文件\n")
    print("开始修复...")
    print("-" * 70)
    
    fixed_count = 0
    for json_path in json_files:
        try:
            fix_sample_file(json_path, original_df)
            fixed_count += 1
        except Exception as e:
            print(f"✗ 修复失败: {json_path.name} - {e}")
    
    print("-" * 70)
    print(f"\n完成！已修复 {fixed_count}/{len(json_files)} 个文件")
    print("\n现在可以重新测试 API，应该会得到正确的预测结果")


if __name__ == "__main__":
    main()

