"""
使用 SHAP 分析模型预测结果，生成特征重要性解释
"""
import json
import pathlib
import joblib
import pandas as pd
# import numpy as np
# import lightgbm as lgb
import shap

# 使用Project_A的模型文件
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT_A_ROOT = PROJECT_ROOT.parent / "Project_A"
MODEL_DIR = PROJECT_A_ROOT / "models"
DATA_DIR = PROJECT_A_ROOT / "data" / "processed"


def load_model():
    """加载模型和预处理器"""
    model_path = MODEL_DIR / "lgbm_model.pkl"
    scaler_path = DATA_DIR / "scaler.joblib"
    columns_path = DATA_DIR / "columns.json"
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    with open(columns_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    
    return model, scaler, feature_names


def explain_prediction(model, scaler, feature_names, sample_data: dict):
    """
    解释单条样本的预测结果
    
    Args:
        model: 训练好的模型
        scaler: 标准化器
        feature_names: 特征列名列表
        sample_data: 单条样本的特征字典
    
    Returns:
        dict: 包含预测结果、特征重要性等信息
    """
    # 转换为DataFrame
    df = pd.DataFrame([sample_data])
    df = df[feature_names]
    
    # 标准化
    X_scaled = scaler.transform(df)
    
    # 预测
    try:
        # 尝试使用predict_proba（sklearn包装的模型）
        pred_proba = model.predict_proba(X_scaled)[0]
        pred_score = pred_proba[1]  # 欺诈概率
    except AttributeError:
        # 如果是Booster对象，predict返回概率（二分类任务）
        pred_score = model.predict(X_scaled)[0]
    
    # SHAP解释
    # TreeExplainer默认使用tree_path_dependent模式，只支持model_output="raw"
    # raw模式返回的是log-odds空间的SHAP值，这是正常的
    explainer = shap.TreeExplainer(model)
    
    # SHAP需要2维数组，使用X_scaled而不是X_scaled[0]
    shap_values = explainer.shap_values(X_scaled)
    
    # 处理SHAP值（二分类返回列表）
    if isinstance(shap_values, list):
        # 二分类：shap_values是列表，[0]是负类，[1]是正类
        shap_values = shap_values[1][0]  # 取正类的SHAP值，然后取第一个样本
    else:
        # 单输出
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # 取第一个样本
        else:
            shap_values = shap_values  # 已经是1维
    
    # 获取特征重要性（按绝对值排序）
    feature_importance = []
    for i, feat_name in enumerate(feature_names):
        feature_importance.append({
            "feature": feat_name,
            "shap_value": float(shap_values[i]),
            "value": float(X_scaled[0][i]),
            "abs_shap": abs(float(shap_values[i]))
        })
    
    # 按绝对值排序
    feature_importance.sort(key=lambda x: x["abs_shap"], reverse=True)
    
    # 获取最重要的前10个特征
    top_features = feature_importance[:10]
    
    return {
        "prediction_score": float(pred_score),
        "prediction_label": int(pred_score >= 0.5),
        "top_features": top_features,
        "all_features": feature_importance
    }


def generate_explanation_text(explanation: dict) -> str:
    """
    生成结构化的解释文本，供LLM使用
    
    Args:
        explanation: explain_prediction返回的字典
    
    Returns:
        str: 格式化的解释文本
    """
    score = explanation["prediction_score"]
    label = "欺诈" if explanation["prediction_label"] == 1 else "正常"
    top_features = explanation["top_features"]
    
    text = f"预测结果：{label}（风险分数：{score:.4f}）\n\n"
    text += "主要影响因素（Top 10）：\n"
    
    for i, feat in enumerate(top_features, 1):
        direction = "增加风险" if feat["shap_value"] > 0 else "降低风险"
        text += f"{i}. {feat['feature']}: {feat['shap_value']:.4f} ({direction})\n"
    
    return text


if __name__ == "__main__":
    # 测试
    model, scaler, feature_names = load_model()
    
    # 示例数据
    sample = {
        "Time": 0.0,
        "V1": -1.3598,
        "V2": -0.0727,
        "V3": 2.5363,
        "V4": 1.3782,
        "V5": -0.3383,
        "V6": 0.4624,
        "V7": 0.2396,
        "V8": 0.0987,
        "V9": 0.3637,
        "V10": 0.0908,
        "V11": -0.5516,
        "V12": -0.6178,
        "V13": -0.9914,
        "V14": -0.3112,
        "V15": 1.4682,
        "V16": -0.4704,
        "V17": 0.2079,
        "V18": 0.0258,
        "V19": 0.4039,
        "V20": 0.2514,
        "V21": -0.0183,
        "V22": 0.2778,
        "V23": -0.1105,
        "V24": 0.0669,
        "V25": 0.1285,
        "V26": -0.1891,
        "V27": 0.1336,
        "V28": -0.0211,
        "Amount": 149.62
    }
    
    result = explain_prediction(model, scaler, feature_names, sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    print("\n" + "="*50)
    print(generate_explanation_text(result))

