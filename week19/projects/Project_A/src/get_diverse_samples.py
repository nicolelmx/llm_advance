"""
获取不同score值的样本（包括中间值）
"""
import json
import pathlib
import pandas as pd
import joblib
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "sample_data"


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


def predict_batch(model, scaler, feature_names, df):
    """批量预测"""
    X = df[feature_names].values
    X_scaled = scaler.transform(X)
    
    try:
        proba = model.predict_proba(X_scaled)
        scores = proba[:, 1]  # 欺诈概率
    except AttributeError:
        proba = model.predict(X_scaled)
        scores = proba
    
    return scores


def find_closest_samples(scores, target_scores, df, n_samples=1):
    """找到最接近目标score的样本"""
    logger.info(f"find_closest_samples start, scores={scores}, target_scores={target_scores}, n_samples={n_samples}")
    results = []
    
    for target_score in target_scores:
        # 找到最接近目标score的样本
        diff = np.abs(scores - target_score)
        closest_indices = np.argsort(diff)[:n_samples]
        
        samples = []
        for idx in closest_indices:
            sample = df.iloc[idx]
            features = sample.drop('Class').to_dict()
            actual_score = scores[idx]
            
            samples.append({
                'features': features,
                'predicted_score': float(actual_score),
                'true_label': int(sample['Class']),
                'predicted_label': int(actual_score >= 0.5),
                'target_score': target_score,
                'difference': float(abs(actual_score - target_score))
            })
        
        results.append({
            'target_score': target_score,
            'samples': samples
        })

    logger.info(f"find_closest_samples start, scores={scores}, target_scores={target_scores}, n_samples={n_samples}")
    return results


def main():
    """主函数"""
    print("=" * 70)
    print("获取不同score值的样本")
    print("=" * 70)
    
    # 加载模型和数据
    print("\n加载模型和数据...")
    model, scaler, feature_names = load_model_and_scaler()
    
    # 加载训练集和测试集
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"总样本数: {len(all_df)}")
    
    # 预测所有样本（使用采样以加快速度）
    print("\n预测样本（使用采样以加快速度）...")
    sample_size = min(50000, len(all_df))  # 最多采样5万个样本
    sampled_df = all_df.sample(n=sample_size, random_state=42)
    scores = predict_batch(model, scaler, feature_names, sampled_df)
    
    print(f"采样样本数: {len(sampled_df)}")
    print(f"\nScore分布统计:")
    print(f"  最小值: {np.min(scores):.6f}")
    print(f"  最大值: {np.max(scores):.6f}")
    print(f"  平均值: {np.mean(scores):.6f}")
    print(f"  中位数: {np.median(scores):.6f}")
    
    # 定义目标score值
    target_scores = [
        1.0,    # 极高风险
        0.9,    # 很高风险
        0.7,    # 高风险
        0.5,    # 中等风险（阈值）
        0.3,    # 低风险
        0.1,    # 很低风险
        0.05,   # 极低风险
        0.01,   # 非常低风险
        0.0     # 无风险
    ]
    
    print("\n" + "=" * 70)
    print("查找不同score值的样本")
    print("=" * 70)
    
    # 查找最接近目标score的样本
    results = find_closest_samples(scores, target_scores, sampled_df, n_samples=1)
    
    all_samples = {}
    
    for result in results:
        target = result['target_score']
        samples = result['samples']
        
        if samples:
            sample = samples[0]
            level_name = f"score_{target:.2f}".replace('.', '_')
            all_samples[level_name] = [sample]
            
            print(f"\n目标score: {target:.2f}")
            print(f"  实际score: {sample['predicted_score']:.6f}")
            print(f"  差异: {sample['difference']:.6f}")
            print(f"  真实标签: {sample['true_label']} ({'欺诈' if sample['true_label'] == 1 else '正常'})")
            print(f"  预测标签: {sample['predicted_label']} ({'欺诈' if sample['predicted_label'] == 1 else '正常'})")
            print(f"  风险等级: ", end="")
            
            if target >= 0.9:
                print("极高风险")
            elif target >= 0.7:
                print("高风险")
            elif target >= 0.5:
                print("中等风险")
            elif target >= 0.1:
                print("低风险")
            else:
                print("极低风险")
    
    # 保存样本
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("\n" + "=" * 70)
    print("保存样本文件")
    print("=" * 70)
    
    for level_name, samples in all_samples.items():
        for i, sample in enumerate(samples, 1):
            filename = OUTPUT_DIR / f"{level_name}_sample.json"
            output_data = {
                "target_score": sample['target_score'],
                "actual_score": sample['predicted_score'],
                "true_label": sample['true_label'],
                "predicted_label": sample['predicted_label'],
                "risk_level": "",
                "features": sample['features']
            }
            
            # 添加风险等级描述
            score = sample['predicted_score']
            if score >= 0.9:
                output_data['risk_level'] = "极高风险"
            elif score >= 0.7:
                output_data['risk_level'] = "高风险"
            elif score >= 0.5:
                output_data['risk_level'] = "中等风险"
            elif score >= 0.1:
                output_data['risk_level'] = "低风险"
            else:
                output_data['risk_level'] = "极低风险"
            
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"  已保存: {filename.name} (score={sample['predicted_score']:.6f})")
    
    # 创建汇总文件
    summary_file = OUTPUT_DIR / "diverse_samples_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"\n汇总文件: {summary_file.name}")
    
    # 生成使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print(f"\n样本文件保存在: {OUTPUT_DIR}")
    print("\n可用的样本文件：")
    for level_name in sorted(all_samples.keys()):
        sample = all_samples[level_name][0]
        print(f"  - {level_name}_sample.json")
        print(f"    实际score: {sample['predicted_score']:.6f}, 风险等级: {sample['predicted_label']}")
    
    print("\nAPI测试步骤：")
    print("  1. 打开 http://localhost:8000/docs")
    print("  2. 选择 /predict 接口")
    print("  3. 点击 'Try it out'")
    print("  4. 复制JSON文件中的 'features' 部分")
    print("  5. 粘贴到 Request body 中")
    print("  6. 点击 'Execute' 查看结果")
    
    print("\n预期结果：")
    print("  score_1_00: score应该接近1.0, label=1")
    print("  score_0_90: score应该接近0.9, label=1")
    print("  score_0_70: score应该接近0.7, label=1")
    print("  score_0_50: score应该接近0.5, label=1或0")
    print("  score_0_30: score应该接近0.3, label=0")
    print("  score_0_10: score应该接近0.1, label=0")
    print("  score_0_00: score应该接近0.0, label=0")


if __name__ == "__main__":
    main()

