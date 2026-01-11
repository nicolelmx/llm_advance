"""
获取高风险样本（欺诈样本）用于API测试
"""
import json
import pathlib
import pandas as pd
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
ORIGINAL_DATA = PROJECT_ROOT / "data" / "creditcard.csv"


def get_high_risk_samples_from_processed():
    """从预处理后的测试集中获取高风险样本"""
    print("=" * 60)
    print("方法1：从预处理后的测试集中获取高风险样本")
    print("=" * 60)
    
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    
    # 筛选欺诈样本（Class=1）
    fraud_samples = test_df[test_df["Class"] == 1]
    
    if len(fraud_samples) > 0:
        print(f"[OK] 找到 {len(fraud_samples)} 个欺诈样本\n")
        
        # 选择第一个欺诈样本
        sample = fraud_samples.iloc[0]
        
        # 提取特征（排除Class）
        features = sample.drop("Class").to_dict()
        
        print("高风险样本（欺诈样本）：")
        print("-" * 60)
        print(f"Class (真实标签): {sample['Class']}")
        print(f"特征数量: {len(features)}")
        print("\n特征值（前10个）：")
        for i, (key, value) in enumerate(list(features.items())[:10]):
            print(f"  {key}: {value:.4f}")
        print("  ...")
        
        return features, sample['Class']
    else:
        print("[X] 测试集中没有欺诈样本")
        return None, None


def get_high_risk_samples_from_original():
    """从原始数据集中获取高风险样本"""
    print("\n" + "=" * 60)
    print("方法2：从原始数据集中获取高风险样本")
    print("=" * 60)
    
    df = pd.read_csv(ORIGINAL_DATA)
    
    # 筛选欺诈样本
    fraud_samples = df[df["Class"] == 1]
    
    if len(fraud_samples) > 0:
        print(f"[OK] 找到 {len(fraud_samples)} 个欺诈样本\n")
        
        # 选择第一个欺诈样本
        sample = fraud_samples.iloc[0]
        
        # 提取特征（排除Class）
        features = sample.drop("Class").to_dict()
        
        print("高风险样本（原始数据，需要标准化）：")
        print("-" * 60)
        print(f"Class (真实标签): {sample['Class']}")
        print(f"Time: {sample['Time']}")
        print(f"Amount: {sample['Amount']}")
        print(f"特征数量: {len(features)}")
        
        return features, sample['Class']
    else:
        print("[X] 原始数据集中没有欺诈样本")
        return None, None


def construct_high_risk_sample():
    """根据特征重要性构造高风险样本"""
    print("\n" + "=" * 60)
    print("方法3：构造高风险样本（基于特征重要性）")
    print("=" * 60)
    
    # 根据已知的特征重要性，构造高风险特征值
    # V14通常是降低风险的特征（负值降低风险，所以高风险应该是正值）
    # V4、V3等正值通常增加风险
    
    high_risk_features = {
        "Time": 0.0,
        "V1": 2.0,      # 较大的正值，增加风险
        "V2": 1.5,      # 正值，增加风险
        "V3": 3.0,      # 较大的正值，增加风险
        "V4": 2.5,      # 较大的正值，增加风险
        "V5": 1.0,      # 正值，增加风险
        "V6": 1.2,      # 正值，增加风险
        "V7": 1.5,      # 正值，增加风险
        "V8": 0.5,      # 较小的正值
        "V9": 1.0,      # 正值，增加风险
        "V10": 1.2,     # 正值，增加风险
        "V11": 1.5,     # 正值，增加风险
        "V12": 1.0,     # 正值，增加风险
        "V13": 1.2,     # 正值，增加风险
        "V14": 1.0,     # 正值（注意：V14通常是负值降低风险，但这里用正值测试）
        "V15": 1.5,     # 正值，增加风险
        "V16": 1.0,     # 正值，增加风险
        "V17": 0.8,     # 正值，增加风险
        "V18": 0.5,     # 较小的正值
        "V19": 1.0,     # 正值，增加风险
        "V20": 1.2,     # 正值，增加风险
        "V21": 0.5,     # 较小的正值
        "V22": 1.0,     # 正值，增加风险
        "V23": 0.5,     # 较小的正值
        "V24": 0.8,     # 正值，增加风险
        "V25": 0.5,     # 较小的正值
        "V26": 0.3,     # 较小的正值
        "V27": 1.0,     # 正值，增加风险
        "V28": 0.2,     # 较小的正值
        "Amount": 500.0  # 较大的金额，可能增加风险
    }
    
    print("构造的高风险样本特征值：")
    print("-" * 60)
    print("注意：这是构造的样本，用于测试高风险情况")
    print("特征值特点：")
    print("  - V1-V28大部分为正值（增加风险）")
    print("  - Amount较大（500）")
    print("  - 这些特征组合应该产生较高的风险分数")
    
    return high_risk_features, None


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("获取高风险样本用于API测试")
    print("=" * 60)
    
    # 方法1：从预处理后的数据获取（推荐）
    features, label = get_high_risk_samples_from_processed()
    
    if features is None:
        # 方法2：从原始数据获取
        features, label = get_high_risk_samples_from_original()
    
    if features is None:
        # 方法3：构造高风险样本
        features, label = construct_high_risk_sample()
    
    # 生成JSON格式
    json_data = {"features": features}
    
    print("\n" + "=" * 60)
    print("JSON格式（可直接用于API测试）")
    print("=" * 60)
    print(json.dumps(json_data, indent=2))
    
    # 保存到文件
    output_file = PROJECT_ROOT / "high_risk_sample.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] 已保存到: {output_file}")
    
    if label is not None:
        print(f"\n真实标签: {label} (1=欺诈)")
    
    print("\n" + "=" * 60)
    print("使用说明")
    print("=" * 60)
    print("1. 复制上面的JSON数据")
    print("2. 在API文档页面（http://localhost:8000/docs）测试")
    print("3. 或使用curl命令：")
    print(f'   curl -X POST "http://localhost:8000/predict" \\')
    print(f'     -H "Content-Type: application/json" \\')
    print(f'     -d @{output_file}')
    print("\n预期结果：")
    print("  - score应该 > 0.5（高风险）")
    print("  - label应该 = 1（欺诈）")


if __name__ == "__main__":
    main()

