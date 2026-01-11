import json
import pathlib

import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = pathlib.Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"
OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def split_save(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    y = df["Class"]
    X = df.drop(columns=["Class"])
    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # 标准化数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # fit()得到数据的统计参数
    X_test_scaled = scaler.transform(X_test)  # fit_transform 与 transform的区别？

    # 保存处理后数据与scaler
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df["Class"] = y_train.values
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_df["Class"] = y_test.values

    train_df.to_parquet(OUT_DIR / "train.parquet", index=False)
    test_df.to_parquet(OUT_DIR / "test.parquet", index=False)
    dump(scaler, OUT_DIR / "scaler.joblib")
    with open(OUT_DIR / "columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)
    print("Saved:", OUT_DIR)


def main():
    df = load_data(DATA_PATH)
    split_save(df)


if __name__ == "__main__":
    main()
