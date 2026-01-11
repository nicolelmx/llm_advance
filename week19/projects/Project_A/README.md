# Project A - 风险预测基线（信用卡欺诈示例）

## 目录
- `data/` 数据集（creditcard.csv）
- `src/preprocess.py` 数据清洗与划分
- `src/train_lgbm.py` 训练与保存模型
- `src/infer.py` 本地批量/单条推理
- `src/api/app.py` FastAPI 服务
- `models/` 训练产出的模型文件（pkl/onnx）
- `requirements.txt` 依赖
- `Dockerfile` 容器化运行

## 快速开始
```bash
cd EDU/Project_A
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/preprocess.py
python src/train_lgbm.py
python src/api/app.py  # 启动服务，默认 http://0.0.0.0:8000
```

## Docker 运行
```bash
cd EDU/Project_A
docker build -t risk-api .
docker run -it -p 8000:8000 risk-api
```

## API 示例
```bash
curl -X POST "http://localhost:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"features\": {\"V1\": -1.3598, \"V2\": -0.0727, ... , \"Amount\": 149.62}}"
```

## 说明
- 使用 LightGBM 训练二分类模型，指标 AUC/KS。
- `infer.py` 支持批量 CSV/Parquet 推理。
- 自动导出 ONNX（若失败，查看控制台提示；可安装 `skl2onnx` 并重跑）。
- FastAPI 内置特征列校验、ONNX 优先加载、标准化变换。
- Dockerfile 提供一键容器化运行。

## PowerShell 命令示例（避免续行符问题）
```powershell
python src\infer.py --input data\processed\test.parquet --model models\lgbm_model.pkl --output predictions.parquet
```
（PS 的换行续行符应使用 ``` 后的 ` 而非 \）

