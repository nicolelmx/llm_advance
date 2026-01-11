import os
import pathlib
import joblib
import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import onnxruntime as ort  # 用于加载和运行ONNX格式的模型

MODEL_PATH = pathlib.Path(__file__).resolve().parents[2] / "models" / "lgbm_model.pkl"
MODEL_PATH_ONNX = pathlib.Path(__file__).resolve().parents[2] / "models" / "lgbm_model.onnx"
SCALER_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "scaler.joblib"
COLUMNS_PATH = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "columns.json"

app = FastAPI(title="Risk Prediction API", version="0.1.0")


class Features(BaseModel):
    features: dict


def load_model():
    # 优先加载 ONNX，如不存在则加载 pkl
    if MODEL_PATH_ONNX.exists():
        session = ort.InferenceSession(str(MODEL_PATH_ONNX))
        return ("onnx", session)
    if MODEL_PATH.exists():
        if MODEL_PATH.suffix == ".pkl":
            return ("lgbm", joblib.load(MODEL_PATH))
        return ("lgbm_txt", lgb.Booster(model_file=str(MODEL_PATH)))
    raise FileNotFoundError("No model found (onnx or pkl).")


def load_scaler():
    if not SCALER_PATH.exists():
        return None
    return joblib.load(SCALER_PATH)


def load_columns():
    if not COLUMNS_PATH.exists():
        return None
    with open(COLUMNS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


model = load_model()
scaler = load_scaler()
feature_names = load_columns()


@app.post("/predict")
def predict(payload: Features):  # 如果说Features验证失败，会去报422的错误
    try:
        df = pd.DataFrame([payload.features])  # 为什么要把字典包装成列表？
        # 列校验与重排
        if feature_names:
            missing = set(feature_names) - set(df.columns)
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
            df = df[feature_names]

        # 标准化特征值
        # 注意：如果输入的特征值已经是标准化后的（如来自 sample_data），
        # 这里会再次标准化，导致预测错误。
        # 解决方案：API 期望输入原始值（未标准化的），或者修改样本生成脚本保存原始值
        if scaler is not None:
            # scaler.transform() 返回 numpy array，需要正确赋值
            X_scaled = scaler.transform(df.values)
            df = pd.DataFrame(X_scaled, columns=df.columns, index=df.index)
        if model[0] == "onnx":
            input_name = model[1].get_inputs()[0].name
            proba = model[1].run(None, {input_name: df.values.astype("float32")})[0].ravel() # 展平为一维数组
            # ONNX模型可能返回单个值或概率数组
            if len(proba) == 1:
                score = float(proba[0])
            else:
                # 如果是二分类的概率数组，取正类概率
                score = float(proba[1]) if len(proba) > 1 else float(proba[0])
        elif model[0] == "lgbm":
            # joblib保存的模型，可能是sklearn包装的或Booster对象
            try:
                # 尝试使用predict_proba（sklearn包装的模型）
                proba = model[1].predict_proba(df.values)
                if len(proba.shape) == 1:
                    score = float(proba[1]) if len(proba) > 1 else float(proba[0])
                else:
                    score = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            except AttributeError:
                # 如果是Booster对象，predict返回概率（二分类任务）
                proba = model[1].predict(df.values)
                score = float(proba[0])
        else:
            # lgb.Booster对象，predict方法返回概率（对于二分类）
            proba = model[1].predict(df.values)
            score = float(proba[0])
        return {"score": score, "label": int(score >= 0.5)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Risk Prediction API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)"
        },
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
