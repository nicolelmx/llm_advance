"""
Project_B API服务：整合LightGBM预测和Ollama解释
"""
import os
import pathlib
import sys

from week19.projects.Project_B.src.explain import load_model, explain_prediction, generate_explanation_text
from week19.projects.Project_B.src.ollama_client import OllamaClient

# from week19.更新后代码.Project_B.src.explain import load_model

# 添加项目根目录到路径
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import json
import requests

# 导入自定义模块
# from ..explain import load_model, explain_prediction, generate_explanation_text
# from ..ollama_client import OllamaClient

# 路径配置（使用Project_A的模型）
PROJECT_A_ROOT = PROJECT_ROOT.parent / "Project_A"
MODEL_DIR = PROJECT_A_ROOT / "models"
DATA_DIR = PROJECT_A_ROOT / "data" / "processed"

app = FastAPI(
    title="智能风控预测与解释API",
    version="0.2.0",
    description="""
    ## 智能风控预测与解释系统 API
    
    本API提供以下功能：
    
    * **欺诈预测**：使用LightGBM模型预测信用卡交易是否为欺诈
    * **SHAP解释**：分析哪些特征对预测结果影响最大
    * **LLM解释**：使用qwen3:4b生成通俗易懂的中文解释报告
    * **策略建议**：针对高风险交易生成风控策略建议
    
    ### 使用说明
    
    1. 确保Ollama服务正在运行（默认端口8001）
    2. 确保Project_A的模型文件已准备好
    3. 使用 `/predict` 接口进行预测和解释
    4. 查看 `/docs` 获取完整的API文档
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=[
        {
            "name": "预测",
            "description": "欺诈预测相关接口"
        },
        {
            "name": "解释",
            "description": "模型解释相关接口"
        },
        {
            "name": "健康检查",
            "description": "服务健康检查接口"
        }
    ]
)

# 全局变量
model = None
scaler = None
feature_names = None
ollama_client = None


class PredictionRequest(BaseModel):
    """预测请求模型"""
    features: Dict[str, float] = Field(..., description="30个特征值（Time, V1-V28, Amount）")
    explain: bool = Field(True, description="是否生成SHAP解释")
    use_llm: bool = Field(False, description="是否使用LLM生成解释报告（默认false，因为可能需要10-30秒）")
    model_name: str = Field("qwen3:4b", description="Ollama模型名称")
    scene_img: UploadFile = File(None, description="交易场景图片（商户/终端截图）")

    class Config:
        schema_extra = {
            "example": {
                "features": {
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
                },
                "explain": True,
                "use_llm": True,
                "model_name": "qwen3:4b"
            }
        }


class PredictionResponse(BaseModel):
    """预测响应模型"""
    score: float = Field(..., description="风险分数（0-1之间，越高越危险）")
    label: int = Field(..., description="预测标签（0=正常，1=欺诈）")
    explanation: Optional[Dict] = Field(None, description="SHAP特征重要性解释")
    llm_explanation: Optional[str] = Field(None, description="LLM生成的中文解释报告")
    strategy_suggestion: Optional[str] = Field(None, description="风控策略建议（仅高风险交易）")


@app.on_event("startup")
async def startup_event():
    """应用启动时加载模型和初始化客户端"""
    global model, scaler, feature_names, ollama_client
    
    try:
        # 加载模型
        model, scaler, feature_names = load_model()
        print("✓ 模型加载成功")
        
        # 初始化Ollama客户端
        ollama_client = OllamaClient()
        if ollama_client.health_check():
            print("✓ Ollama服务连接成功")
        else:
            print("⚠ Ollama服务不可用，LLM解释功能将不可用")
            ollama_client = None
    except Exception as e:
        print(f"✗ 启动失败: {e}")
        raise


@app.get("/", tags=["健康检查"])
def root():
    """根路径 - 获取API基本信息"""
    return {
        "message": "智能风控预测与解释API",
        "version": "0.2.0",
        "description": "整合LightGBM预测和Ollama LLM解释的智能风控系统",
        "endpoints": {
            "health": "/health - 服务健康检查",
            "predict": "/predict (POST) - 预测接口（带解释）",
            "explain": "/explain (POST) - 仅生成解释",
            "ollama_health": "/ollama/health - Ollama服务检查"
        },
        "docs": {
            "swagger": "/docs - Swagger UI文档",
            "redoc": "/redoc - ReDoc文档"
        }
    }


@app.get("/health", tags=["健康检查"])
def health():
    """服务健康检查 - 检查模型和Ollama服务状态"""
    return {
        "status": "ok",
        "message": "服务运行正常",
        "model_loaded": model is not None,
        "ollama_available": ollama_client is not None if ollama_client else False
    }


@app.get("/ollama/health", tags=["健康检查"])
def ollama_health():
    """Ollama服务健康检查 - 检查Ollama服务是否可用"""
    if ollama_client is None:
        raise HTTPException(status_code=503, detail="Ollama客户端未初始化")
    
    is_healthy = ollama_client.health_check()
    return {
        "status": "ok" if is_healthy else "unavailable",
        "message": "Ollama服务可用" if is_healthy else "Ollama服务不可用",
        "base_url": ollama_client.base_url
    }


@app.post("/predict", response_model=PredictionResponse, tags=["预测"])
def predict(request: PredictionRequest):
    """
    欺诈预测接口（带解释功能）
    
    **功能说明：**
    - 接收30个特征值（Time, V1-V28, Amount）
    - 使用LightGBM模型进行欺诈预测
    - 可选：生成SHAP特征重要性解释
    - 可选：使用Ollama LLM生成中文解释报告
    - 高风险交易自动生成策略建议
    
    **参数说明：**
    - `features`: 30个特征值字典（必需）
    - `explain`: 是否生成SHAP解释（默认true）
    - `use_llm`: 是否使用LLM生成解释（默认false，因为可能需要10-30秒）
    - `model_name`: Ollama模型名称（默认"qwen3:4b"）
    
    **返回说明：**
    - `score`: 风险分数（0-1之间）
    - `label`: 预测标签（0=正常，1=欺诈）
    - `explanation`: SHAP特征重要性分析
    - `llm_explanation`: LLM生成的中文解释
    - `strategy_suggestion`: 策略建议（仅高风险）
    
    **注意：**
    - LLM解释生成可能需要10-30秒，请耐心等待
    - 如果Ollama服务不可用，将只返回SHAP解释
    - 默认 `use_llm=false`，如需LLM解释请手动设置为 `true`
    """
    try:
        # 验证特征
        if feature_names:
            missing = set(feature_names) - set(request.features.keys())
            if missing:
                raise ValueError(f"Missing features: {sorted(missing)}")
        
        # 预测
        explanation_data = explain_prediction(
            model, scaler, feature_names, request.features
        )
        
        score = explanation_data["prediction_score"]
        label = explanation_data["prediction_label"]
        
        response = PredictionResponse(
            score=score,
            label=label
        )
        
        # 生成解释
        if request.explain:
            response.explanation = {
                "top_features": explanation_data["top_features"],
                "explanation_text": generate_explanation_text(explanation_data)
            }
            
            # 使用LLM生成解释
            if request.use_llm and ollama_client:
                try:
                    explanation_text = response.explanation["explanation_text"]
                    print(f"[DEBUG] 开始生成LLM解释...")
                    llm_explanation = ollama_client.explain_prediction(
                        model=request.model_name,
                        explanation_text=explanation_text
                    )
                    print(f"[DEBUG] LLM解释生成完成")
                    response.llm_explanation = llm_explanation
                    
                    # 生成策略建议
                    if label == 1:  # 如果是欺诈，生成策略建议
                        print(f"[DEBUG] 开始生成策略建议...")
                        strategy = ollama_client.generate_strategy(
                            model=request.model_name,
                            risk_summary=explanation_text
                        )
                        print(f"[DEBUG] 策略建议生成完成")
                        response.strategy_suggestion = strategy
                except requests.exceptions.Timeout:
                    print(f"[WARNING] LLM解释生成超时")
                    response.llm_explanation = "LLM解释生成超时，请稍后重试或检查Ollama服务"
                except requests.exceptions.ConnectionError:
                    print(f"[WARNING] 无法连接到Ollama服务")
                    response.llm_explanation = "无法连接到Ollama服务，请检查服务是否运行"
                except Exception as e:
                    print(f"[ERROR] LLM解释生成失败: {e}")
                    import traceback
                    traceback.print_exc()
                    response.llm_explanation = f"LLM解释生成失败: {str(e)}"
                    # 不抛出异常，继续返回基础解释
        
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/explain", tags=["解释"])
def explain_only(request: PredictionRequest):
    """
    仅生成解释接口（不返回预测结果）
    
    **功能说明：**
    - 仅对输入特征进行SHAP分析和LLM解释
    - 不返回预测分数和标签
    - 适用于已知道预测结果，只需要解释的场景
    
    **参数说明：**
    与 `/predict` 接口相同
    """
    try:
        explanation_data = explain_prediction(
            model, scaler, feature_names, request.features
        )
        
        result = {
            "explanation": {
                "top_features": explanation_data["top_features"],
                "explanation_text": generate_explanation_text(explanation_data)
            }
        }
        
        if request.use_llm and ollama_client:
            try:
                llm_explanation = ollama_client.explain_prediction(
                    model=request.model_name,
                    explanation_text=result["explanation"]["explanation_text"]
                )
                result["llm_explanation"] = llm_explanation
            except Exception as e:
                print(f"LLM解释生成失败: {e}")
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8001))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

