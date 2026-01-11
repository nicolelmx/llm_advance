"""
生成一条示例样本的 JSON（用于 /predict 请求）

用法：
    python src/sample_json.py
运行后会在终端打印一条包含 30 个特征键的 JSON。
"""

import json
import pathlib

import pandas as pd


def main():
    p = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "test.parquet"
    df = pd.read_parquet(p)
    row = df.drop(columns=["Class"]).iloc[0].to_dict()
    print(json.dumps({"features": row}, indent=2))


if __name__ == "__main__":
    main()

