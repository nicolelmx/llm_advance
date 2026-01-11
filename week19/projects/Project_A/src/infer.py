import json
import pathlib
import argparse
import joblib
import pandas as pd
import lightgbm as lgb
import onnxruntime as ort

DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
DEFAULT_MODEL = MODEL_DIR / "lgbm_model.pkl"
DEFAULT_ONNX = MODEL_DIR / "lgbm_model.onnx"


def load_model(path: pathlib.Path):
    if path.suffix == ".pkl":
        return ("lgbm", joblib.load(path))
    if path.suffix == ".onnx":
        session = ort.InferenceSession(str(path))
        return ("onnx", session)
    model = lgb.Booster(model_file=str(path))
    return ("lgbm_txt", model)


def run_infer(input_path: pathlib.Path, model_path: pathlib.Path):
    # load feature order
    with open(DATA_DIR / "columns.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    df = pd.read_parquet(input_path)
    if "Class" in df.columns:
        df = df.drop(columns=["Class"])  # 防止数据泄露
    df = df[feature_names]

    model_type, model = load_model(model_path)
    if model_type == "onnx":
        input_name = model.get_inputs()[0].name
        preds = model.run(None, {input_name: df.values.astype("float32")})[0].ravel()
    else:
        preds = model.predict(df.values)

    out = df.copy()
    out["pred"] = preds
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=str(DATA_DIR / "test.parquet"))
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--output", type=str, default="predictions.parquet")
    args = parser.parse_args()

    result = run_infer(pathlib.Path(args.input), pathlib.Path(args.model))
    out_path = pathlib.Path(args.output)
    result.to_parquet(out_path, index=False)
    print("Saved predictions to", out_path.resolve())


if __name__ == "__main__":
    main()
