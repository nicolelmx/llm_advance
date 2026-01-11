import json
import pathlib
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import time
import os
import shutil


def get_short_path(long_path):
    """
    获取 Windows 短路径名（8.3 格式），用于解决中文路径编码问题
    """
    try:
        import win32api
        return win32api.GetShortPathName(str(long_path))
    except ImportError:
        # 如果没有 win32api，尝试使用 ctypes
        try:
            import ctypes
            from ctypes import wintypes
            _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            _GetShortPathNameW.restype = wintypes.DWORD
            
            path_str = str(long_path)
            buffer = ctypes.create_unicode_buffer(260)
            length = _GetShortPathNameW(path_str, buffer, 260)
            if length > 0:
                return buffer.value
        except Exception:
            pass
    return str(long_path)


DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    # 特征列顺序
    with open(DATA_DIR / "columns.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    y_train = train_df["Class"].values
    X_train = train_df[feature_names].values
    y_test = test_df["Class"].values
    X_test = test_df[feature_names].values
    return X_train, y_train, X_test, y_test, feature_names


def train_lgbm(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True,
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def main():
    X_train, y_train, X_test, y_test, feature_names = load_data()
    model = train_lgbm(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    # 保存模型文件
    model_path = MODEL_DIR / "lgbm_model.txt"
    pkl_path = MODEL_DIR / "lgbm_model.pkl"
    
    # 如果文件已存在，先删除（避免文件被占用的问题）
    if model_path.exists():
        try:
            model_path.unlink()
            print(f"Removed existing model file: {model_path}")
            # 等待文件系统完全释放文件句柄
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not remove existing file {model_path}: {e}")
            print("Please close any programs that might be using this file and try again.")
            return
    
    if pkl_path.exists():
        try:
            pkl_path.unlink()
            print(f"Removed existing pkl file: {pkl_path}")
            time.sleep(0.1)
        except Exception as e:
            print(f"Warning: Could not remove existing file {pkl_path}: {e}")
    
    # 保存模型 - 使用多种方法避免中文路径编码问题
    model_saved = False
    
    # 方法1：尝试使用短路径名（8.3 格式）
    short_path = get_short_path(model_path)
    if short_path != str(model_path):
        print(f"Trying short path method: {short_path}")
        try:
            model.save_model(short_path)
            # 如果保存到短路径成功，但目标路径不同，需要复制
            if short_path != str(model_path):
                if model_path.exists():
                    model_path.unlink()
                shutil.copy2(short_path, str(model_path))
            print("Saved model to:", model_path)
            model_saved = True
        except Exception as e1:
            print(f"Short path method failed: {e1}")
    
    # 方法2：如果短路径方法失败，尝试直接保存
    if not model_saved:
        model_str_path = str(model_path)
        try:
            model.save_model(model_str_path)
            print("Saved model to:", model_path)
            model_saved = True
        except Exception as e2:
            print(f"Direct save failed: {e2}")
    
    # 方法3：如果前两种方法都失败，使用临时文件方法
    if not model_saved:
        print("Trying temporary file method...")
        try:
            import tempfile
            # 使用系统临时目录（通常没有中文路径）
            temp_dir = tempfile.gettempdir()
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt', dir=temp_dir) as tmp_file:
                tmp_path = tmp_file.name
            
            # 保存到临时文件
            model.save_model(tmp_path)
            # 删除目标文件（如果存在）
            if model_path.exists():
                model_path.unlink()
                time.sleep(0.1)
            # 复制临时文件到目标位置
            shutil.copy2(tmp_path, str(model_path))
            # 删除临时文件
            os.unlink(tmp_path)
            print("Saved model to (via temp file):", model_path)
            model_saved = True
        except Exception as e3:
            print(f"Temporary file method also failed: {e3}")
            print(f"\nError saving model to {model_path}")
            print("Possible causes:")
            print("1. File is being used by another process")
            print("2. Insufficient permissions")
            print("3. Path encoding issue with Chinese characters")
            print("\nWorkaround: Try moving the project to a path without Chinese characters")
            raise Exception("All save methods failed. This is likely due to path encoding issues with Chinese characters.")
    
    try:
        joblib.dump(model, str(pkl_path))
        print("Saved pkl model to:", pkl_path)
    except Exception as e:
        print(f"Error saving pkl model: {e}")
        raise

    # 导出 ONNX 便于 CPU 推理/量化后推理
    try:
        # 使用 onnxmltools 自带的 FloatTensorType
        initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=15)
        onnx_path = MODEL_DIR / "lgbm_model.onnx"
        onnxmltools.utils.save_model(onnx_model, str(onnx_path))
        print("Saved ONNX model to:", onnx_path)
    except Exception as e:
        print("ONNX export failed:", e)


if __name__ == "__main__":
    main()

