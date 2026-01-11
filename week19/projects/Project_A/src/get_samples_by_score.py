"""
获取不同风险等级的样本（不同score值）
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


def get_samples_by_score_range(model, scaler, feature_names, test_df, score_min, score_max, count=1):
    """获取指定score范围的样本"""
    # 预测所有样本
    scores = predict_batch(model, scaler, feature_names, test_df)
    
    # 筛选score范围内的样本
    mask = (scores >= score_min) & (scores <= score_max)  # 改为 <= 包含边界
    candidates = test_df[mask].copy()
    candidates['predicted_score'] = scores[mask]
    
    if len(candidates) == 0:
        return None
    
    # 按score排序，选择最接近目标范围的样本
    candidates = candidates.sort_values('predicted_score', ascending=False)
    
    samples = []
    for i in range(min(count, len(candidates))):
        sample = candidates.iloc[i]
        features = sample.drop(['Class', 'predicted_score']).to_dict()
        samples.append({
            'features': features,
            'predicted_score': float(sample['predicted_score']),
            'true_label': int(sample['Class']),
            'predicted_label': int(sample['predicted_score'] >= 0.5)
        })
    
    return samples


def get_samples_by_percentile(model, scaler, feature_names, test_df, percentiles, count=1):
    """按百分位数获取样本"""
    # 预测所有样本
    scores = predict_batch(model, scaler, feature_names, test_df)
    
    # 计算百分位数
    score_percentiles = np.percentile(scores, percentiles)
    
    samples_by_percentile = {}
    for i, p in enumerate(percentiles):
        target_score = score_percentiles[i]
        
        # 找到最接近目标score的样本
        diff = np.abs(scores - target_score)
        closest_indices = np.argsort(diff)[:count]
        
        samples = []
        for idx in closest_indices:
            sample = test_df.iloc[idx]
            features = sample.drop('Class').to_dict()
            samples.append({
                'features': features,
                'predicted_score': float(scores[idx]),
                'true_label': int(sample['Class']),
                'predicted_label': int(scores[idx] >= 0.5),
                'percentile': p
            })
        
        samples_by_percentile[f"p{p}"] = samples
    
    return samples_by_percentile


def main():
    """主函数"""
    print("=" * 70)
    print("获取不同风险等级的样本")
    print("=" * 70)
    
    # 加载模型和数据
    print("\n加载模型和数据...")
    model, scaler, feature_names = load_model_and_scaler()
    
    # 同时加载训练集和测试集，增加样本多样性
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    
    # 合并数据集（只用于查找样本，不用于训练）
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    
    print(f"训练集大小: {len(train_df)}")
    print(f"测试集大小: {len(test_df)}")
    print(f"总样本数: {len(all_df)}")
    print(f"特征数量: {len(feature_names)}")
    
    # 使用合并后的数据集
    test_df = all_df
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 先预测所有样本，分析score分布
    print("\n分析score分布...")
    all_scores = predict_batch(model, scaler, feature_names, test_df)
    
    print(f"Score统计信息:")
    print(f"  最小值: {np.min(all_scores):.6f}")
    print(f"  最大值: {np.max(all_scores):.6f}")
    print(f"  平均值: {np.mean(all_scores):.6f}")
    print(f"  中位数: {np.median(all_scores):.6f}")
    print(f"  25%分位数: {np.percentile(all_scores, 25):.6f}")
    print(f"  75%分位数: {np.percentile(all_scores, 75):.6f}")
    print(f"  95%分位数: {np.percentile(all_scores, 95):.6f}")
    print(f"  99%分位数: {np.percentile(all_scores, 99):.6f}")
    
    # 按百分位数获取样本（更可靠的方法）
    print("\n按百分位数提取样本...")
    percentiles = [99, 95, 90, 75, 50, 25, 10, 5, 1]
    percentile_samples = get_samples_by_percentile(model, scaler, feature_names, test_df, percentiles, count=1)
    
    # 定义不同的风险等级（基于实际分布）
    risk_levels = [
        {
            "name": "极高风险",
            "score_range": (0.9, 1.0),
            "description": "模型非常确信是欺诈，score >= 0.9"
        },
        {
            "name": "高风险",
            "score_range": (0.7, 0.9),
            "description": "高风险，score在0.7-0.9之间"
        },
        {
            "name": "中等风险",
            "score_range": (0.5, 0.7),
            "description": "中等风险，score在0.5-0.7之间"
        },
        {
            "name": "低风险",
            "score_range": (0.1, 0.5),
            "description": "低风险，score在0.1-0.5之间"
        },
        {
            "name": "极低风险",
            "score_range": (0.0, 0.1),
            "description": "极低风险，score < 0.1"
        }
    ]
    
    all_samples = {}
    
    # 添加百分位数样本
    print("\n" + "=" * 70)
    print("按百分位数提取的样本")
    print("=" * 70)
    
    for key, samples in percentile_samples.items():
        if samples:
            percentile = samples[0]['percentile']
            level_name = f"百分位{percentile}"
            all_samples[level_name] = samples
            sample = samples[0]
            print(f"\n{level_name} (score = {sample['predicted_score']:.6f}):")
            print(f"  真实标签: {sample['true_label']} ({'欺诈' if sample['true_label'] == 1 else '正常'})")
            print(f"  预测标签: {sample['predicted_label']} ({'欺诈' if sample['predicted_label'] == 1 else '正常'})")
    
    print("\n" + "=" * 70)
    print("按风险等级提取的样本")
    print("=" * 70)
    
    for level in risk_levels:
        print(f"\n{level['name']} ({level['description']}):")
        print("-" * 70)
        
        samples = get_samples_by_score_range(
            model, scaler, feature_names, test_df,
            level['score_range'][0], level['score_range'][1],
            count=3  # 每个等级取3个样本
        )
        
        if samples:
            all_samples[level['name']] = samples
            
            for i, sample in enumerate(samples, 1):
                print(f"  样本 {i}:")
                print(f"    Score: {sample['predicted_score']:.6f}")
                print(f"    真实标签: {sample['true_label']} ({'欺诈' if sample['true_label'] == 1 else '正常'})")
                print(f"    预测标签: {sample['predicted_label']} ({'欺诈' if sample['predicted_label'] == 1 else '正常'})")
                print(f"    预测正确: {'是' if sample['true_label'] == sample['predicted_label'] else '否'}")
        else:
            print(f"  [X] 未找到score在 {level['score_range']} 范围内的样本")
            # 如果找不到，尝试从百分位数样本中选择最接近的
            if level['name'] == "极高风险":
                # 使用99百分位样本
                if "百分位99" in all_samples:
                    all_samples[level['name']] = all_samples["百分位99"]
                    print(f"  [INFO] 使用99百分位样本代替")
            elif level['name'] == "高风险":
                # 使用95百分位样本
                if "百分位95" in all_samples:
                    all_samples[level['name']] = all_samples["百分位95"]
                    print(f"  [INFO] 使用95百分位样本代替")
            elif level['name'] == "低风险":
                # 使用25百分位样本
                if "百分位25" in all_samples:
                    all_samples[level['name']] = all_samples["百分位25"]
                    print(f"  [INFO] 使用25百分位样本代替")
            elif level['name'] == "极低风险":
                # 使用5百分位样本
                if "百分位5" in all_samples:
                    all_samples[level['name']] = all_samples["百分位5"]
                    print(f"  [INFO] 使用5百分位样本代替")
    
    # 保存所有样本到JSON文件
    print("\n" + "=" * 70)
    print("保存样本到文件")
    print("=" * 70)
    
    for level_name, samples in all_samples.items():
        for i, sample in enumerate(samples, 1):
            filename = OUTPUT_DIR / f"{level_name}_sample_{i}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump({
                    "risk_level": level_name,
                    "sample_index": i,
                    "predicted_score": sample['predicted_score'],
                    "true_label": sample['true_label'],
                    "predicted_label": sample['predicted_label'],
                    "features": sample['features']
                }, f, indent=2, ensure_ascii=False)
            print(f"  已保存: {filename.name}")
    
    # 创建一个汇总文件，包含所有样本
    summary_file = OUTPUT_DIR / "all_samples_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)
    print(f"\n汇总文件: {summary_file.name}")
    
    # 生成使用说明
    print("\n" + "=" * 70)
    print("使用说明")
    print("=" * 70)
    print("\n已生成以下样本文件：")
    print(f"  目录: {OUTPUT_DIR}")
    print("\n文件命名规则：")
    print("  {风险等级}_sample_{序号}.json")
    print("\n风险等级说明：")
    for level in risk_levels:
        print(f"  - {level['name']}: {level['description']}")
    
    print("\nAPI测试示例：")
    print("  1. 在Swagger UI (http://localhost:8000/docs) 中测试")
    print("  2. 复制JSON文件中的features部分")
    print("  3. 粘贴到Request body中")
    print("  4. 点击Execute查看预测结果")
    
    print("\n预期结果对比：")
    print("  极高风险样本: score应该 >= 0.9, label = 1")
    print("  高风险样本:   score应该在 0.7-0.9, label = 1")
    print("  中等风险样本: score应该在 0.5-0.7, label = 1")
    print("  低风险样本:   score应该在 0.1-0.5, label = 0")
    print("  极低风险样本: score应该 < 0.1, label = 0")


if __name__ == "__main__":
    main()

