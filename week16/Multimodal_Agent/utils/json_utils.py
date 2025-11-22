"""
JSON序列化工具 - 处理numpy/pandas类型
"""

import json
import numpy as np
import pandas as pd
from typing import Any, Dict, List

def convert_to_serializable(obj: Any) -> Any:
    """将numpy/pandas类型转换为可JSON序列化的Python原生类型"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def safe_json_dumps(obj: Any, **kwargs) -> str:
    """安全的JSON序列化，自动转换numpy/pandas类型"""
    serializable_obj = convert_to_serializable(obj)
    return json.dumps(serializable_obj, **kwargs)

def safe_json_dump(obj: Any, fp, **kwargs):
    """安全的JSON写入文件，自动转换numpy/pandas类型"""
    serializable_obj = convert_to_serializable(obj)
    json.dump(serializable_obj, fp, **kwargs)

