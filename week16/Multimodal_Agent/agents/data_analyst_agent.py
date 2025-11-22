"""
数据分析师Agent - 多模态数据分析专家
能够理解图表图片、分析CSV数据并生成分析报告
"""

import os
import sys
import logging
import base64
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入基类
from agents.base_agent import BaseAgent
from core.controller import MessageType, TaskStatus, AgentCapability

# 导入工具模块
from utils.file_processor import FileProcessor
from utils.data_quality import DataQualityAnalyzer
from utils.chart_generator import ChartGenerator
from utils.json_utils import convert_to_serializable

# 导入LangChain组件
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataAnalystAgent(BaseAgent):
    """数据分析师Agent - 专门处理数据分析和报告生成"""
    
    def __init__(
        self,
        name: str,
        controller_reference,
        llm_model: str = "gpt-4-vision-preview",  # 支持视觉的多模态模型
        temperature: float = 0.0,
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        # 初始化基类
        capabilities = [
            AgentCapability.IMAGE_PROCESSING,
            AgentCapability.DATA_ANALYSIS,
            AgentCapability.TEXT_PROCESSING,
            AgentCapability.REASONING
        ]
        
        super().__init__(
            name=name,
            capabilities=capabilities,
            controller_reference=controller_reference,
            agent_id=agent_id,
            metadata=metadata or {"description": "Data analyst specialized in chart analysis and CSV data processing"}
        )
        
        # 初始化多模态LLM
        api_key = (os.environ.get("LLM_API_KEY") or 
                  os.environ.get("OPENAI_API_KEY") or 
                  os.environ.get("QWEN_API_KEY"))
        base_url = os.environ.get("LLM_BASE_URL") or os.environ.get("QWEN_BASE_URL")
        
        if not api_key:
            raise ValueError(
                "API key not found. Please set LLM_API_KEY, OPENAI_API_KEY, or QWEN_API_KEY environment variable."
            )
        
        kwargs = {
            "model": llm_model,
            "temperature": temperature,
            "openai_api_key": api_key
        }
        if base_url:
            kwargs["openai_api_base"] = base_url
        
        self.llm = ChatOpenAI(**kwargs)
        
        # 创建分析报告提示模板（将在生成时动态添加用户要求）
        self.base_report_prompt = (
            "你是一位专业的数据分析师。你的任务是：\n"
            "1. 仔细分析提供的图表图片，理解其中的数据趋势、模式和关键信息\n"
            "2. 分析CSV文件中的数据，提取统计信息、趋势和洞察\n"
            "3. 结合图表和CSV数据，生成一份专业、详细的数据分析报告\n"
            "报告应该包括：\n"
            "- 执行摘要\n"
            "- 数据概览\n"
            "- 关键发现\n"
            "- 趋势分析\n"
            "- 结论和建议\n"
            "使用中文撰写报告，确保内容专业、准确、有洞察力。"
        )
        
        logger.info(f"DataAnalystAgent '{name}' initialized")
    
    def execute_task(self, task_info: Dict[str, Any]) -> Any:
        """执行分配的任务"""
        task_type = task_info.get("metadata", {}).get("task_type", "data_analysis")
        
        logger.info(f"DataAnalystAgent '{self.name}' executing task of type: {task_type}")
        
        if task_type == "data_analysis":
            return self._handle_data_analysis_task(task_info)
        elif task_type == "chart_analysis":
            return self._handle_chart_analysis_task(task_info)
        elif task_type == "csv_analysis":
            return self._handle_csv_analysis_task(task_info)
        else:
            logger.warning(f"Unknown task type: {task_type}")
            return {"error": f"Unsupported task type: {task_type}"}
    
    def _handle_chart_analysis_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析图表图片"""
        image_path = task_info.get("metadata", {}).get("image_path")
        image_base64 = task_info.get("metadata", {}).get("image_base64")
        
        if not image_path and not image_base64:
            return {"error": "No image provided for analysis"}
        
        try:
            # 读取图片
            if image_base64:
                # 如果提供了base64编码的图片
                image_data = base64.b64decode(image_base64)
            else:
                # 从文件路径读取
                with open(image_path, "rb") as f:
                    image_data = f.read()
            
            # 转换为base64用于API调用
            image_base64_str = base64.b64encode(image_data).decode('utf-8')
            
            # 使用多模态模型分析图片
            # LangChain支持多模态消息格式
            try:
                # 使用LangChain的多模态消息格式
                from langchain_core.messages import HumanMessage
                
                # 创建包含图片的消息
                messages = [
                    SystemMessage(content=(
                        "你是一位专业的数据分析师，擅长分析各种图表和数据可视化。"
                        "请仔细分析这张图表，描述：\n"
                        "1. 图表的类型和结构\n"
                        "2. 数据的主要趋势和模式\n"
                        "3. 关键数据点和异常值\n"
                        "4. 可能的洞察和结论\n"
                        "请用中文详细描述。"
                    )),
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": "请详细分析这张图表，包括数据趋势、关键发现和洞察。"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64_str}"
                                }
                            }
                        ]
                    )
                ]
                response = self.llm.invoke(messages)
            except Exception as e:
                # 如果多模态格式不支持，尝试使用文本描述
                logger.warning(f"Multimodal format not supported, using text description: {str(e)}")
                messages = [
                    SystemMessage(content=(
                        "你是一位专业的数据分析师，擅长分析各种图表和数据可视化。"
                        "请仔细分析用户提供的图表，描述：\n"
                        "1. 图表的类型和结构\n"
                        "2. 数据的主要趋势和模式\n"
                        "3. 关键数据点和异常值\n"
                        "4. 可能的洞察和结论\n"
                        "请用中文详细描述。"
                    )),
                    HumanMessage(content=(
                        "请详细分析用户上传的图表图片，包括数据趋势、关键发现和洞察。"
                        "图片已上传，请根据图片内容进行分析。"
                    ))
                ]
                response = self.llm.invoke(messages)
            analysis = response.content
            
            logger.info(f"Chart analysis completed")
            
            return {
                "analysis": analysis,
                "image_path": image_path or "base64_image"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing chart: {str(e)}")
            return {
                "error": f"分析图表时出错: {str(e)}",
                "analysis": ""
            }
    
    def _handle_csv_analysis_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """分析CSV文件（支持多种格式）"""
        csv_path = task_info.get("metadata", {}).get("csv_path")
        csv_data = task_info.get("metadata", {}).get("csv_data")
        file_data = task_info.get("metadata", {}).get("file_data")  # bytes格式
        filename = task_info.get("metadata", {}).get("filename", "data.csv")
        
        if not csv_path and not csv_data and not file_data:
            return {"error": "No data provided for analysis"}
        
        try:
            # 使用文件处理器读取数据（支持多种格式）
            if file_data:
                file_info = FileProcessor.read_file(file_data=file_data, filename=filename)
            elif csv_path:
                file_info = FileProcessor.read_file(file_path=csv_path, filename=filename)
            else:
                # 传统CSV方式
                import io
                df = pd.read_csv(io.StringIO(csv_data))
                file_info = {
                    "file_type": "csv",
                    "data": df,
                    "sheets": {"data": df}
                }
            
            if file_info.get("error"):
                return {"error": file_info["error"]}
            
            df = file_info.get("data")
            if df is None or df.empty:
                return {"error": "无法读取数据或数据为空"}
            
            # 数据质量评估
            quality_report = DataQualityAnalyzer.assess_quality(df)
            
            # 异常值检测
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            anomalies = DataQualityAnalyzer.detect_anomalies(df, numeric_cols[:5]) if numeric_cols else {}
            
            # 生成数据摘要
            data_summary = FileProcessor.get_data_summary(df)
            
            # 生成数据统计信息（确保所有值都是可序列化的）
            stats = {
                "shape": [int(df.shape[0]), int(df.shape[1])],  # 转换为Python int
                "columns": df.columns.tolist(),
                "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
                "describe": convert_to_serializable(df.describe().to_dict()) if len(df.select_dtypes(include=['number']).columns) > 0 else {},
                "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
                "sample_data": convert_to_serializable(df.head(10).to_dict('records')),
                "quality_score": float(quality_report.get("overall_score", 0)),
                "anomalies_count": int(anomalies.get("total_anomalies", 0))
            }
            
            # 使用LLM分析数据（包含质量评估）
            # 确保所有数据都是可序列化的
            serializable_stats = convert_to_serializable(stats)
            serializable_quality = convert_to_serializable(quality_report)
            serializable_anomalies = convert_to_serializable(anomalies)
            
            analysis_prompt = (
                "请分析以下数据的统计信息，提供：\n"
                "1. 数据概览（行数、列数、数据类型）\n"
                "2. 数值列的统计摘要\n"
                "3. 缺失值情况\n"
                "4. 数据质量评估（完整性、准确性、一致性）\n"
                "5. 异常值检测结果\n"
                "6. 数据趋势和模式\n"
                "7. 可能的洞察和建议\n\n"
                f"统计信息：\n{json.dumps(serializable_stats, ensure_ascii=False, indent=2)}\n\n"
                f"数据质量评估：\n{json.dumps(serializable_quality, ensure_ascii=False, indent=2)}\n\n"
                f"异常值检测：\n{json.dumps(serializable_anomalies, ensure_ascii=False, indent=2)}\n\n"
                "请用中文详细分析。"
            )
            
            messages = [
                SystemMessage(content="你是一位专业的数据分析师，擅长分析结构化数据。"),
                HumanMessage(content=analysis_prompt)
            ]
            
            response = self.llm.invoke(messages)
            analysis = response.content
            
            logger.info(f"CSV analysis completed for {len(df)} rows")
            
            return {
                "analysis": analysis,
                "statistics": stats,
                "quality_report": quality_report,
                "anomalies": anomalies,
                "data_summary": data_summary,
                "file_type": file_info.get("file_type", "csv"),
                "csv_path": csv_path or filename or "uploaded_data"
            }
            
        except Exception as e:
            logger.error(f"Error analyzing CSV: {str(e)}")
            return {
                "error": f"分析CSV时出错: {str(e)}",
                "analysis": ""
            }
    
    def _handle_data_analysis_task(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """综合分析任务：结合图表和CSV生成报告"""
        image_path = task_info.get("metadata", {}).get("image_path")
        image_base64 = task_info.get("metadata", {}).get("image_base64")
        csv_path = task_info.get("metadata", {}).get("csv_path")
        csv_data = task_info.get("metadata", {}).get("csv_data")
        user_requirements = task_info.get("metadata", {}).get("user_requirements")
        
        chart_analysis = ""
        csv_analysis = ""
        
        # 分析图表
        if image_path or image_base64:
            chart_task = {
                "metadata": {
                    "task_type": "chart_analysis",
                    "image_path": image_path,
                    "image_base64": image_base64
                }
            }
            chart_result = self._handle_chart_analysis_task(chart_task)
            chart_analysis = chart_result.get("analysis", "")
        
        # 处理CSV数据
        # 如果csv_data是字符串且看起来像分析结果（包含中文或较长的文本），直接使用
        # 否则，如果是文件路径或原始数据，则进行分析
        if csv_data:
            # 检查是否是分析结果文本（通常包含中文或较长的文本）
            if isinstance(csv_data, str) and (len(csv_data) > 100 or any('\u4e00' <= char <= '\u9fff' for char in csv_data)):
                # 这看起来是分析结果文本，直接使用
                csv_analysis = csv_data
                logger.info("Using provided CSV analysis result directly")
            elif csv_path:
                # 有文件路径，进行分析
                csv_task = {
                    "metadata": {
                        "task_type": "csv_analysis",
                        "csv_path": csv_path
                    }
                }
                csv_result = self._handle_csv_analysis_task(csv_task)
                csv_analysis = csv_result.get("analysis", "")
            else:
                # 尝试作为原始CSV数据进行分析
                csv_task = {
                    "metadata": {
                        "task_type": "csv_analysis",
                        "csv_data": csv_data
                    }
                }
                csv_result = self._handle_csv_analysis_task(csv_task)
                csv_analysis = csv_result.get("analysis", "")
        elif csv_path:
            # 只有文件路径，进行分析
            csv_task = {
                "metadata": {
                    "task_type": "csv_analysis",
                    "csv_path": csv_path
                }
            }
            csv_result = self._handle_csv_analysis_task(csv_task)
            csv_analysis = csv_result.get("analysis", "")
        
        # 生成综合分析报告
        try:
            # 构建系统提示，包含用户要求
            system_content = self.base_report_prompt
            if user_requirements:
                system_content += f"\n\n**用户特别要求：**\n{user_requirements}\n\n请确保在报告中充分考虑并回应这些要求。"
            
            # 构建用户消息
            user_content = (
                "请基于以下信息生成数据分析报告：\n\n"
                "图表分析：\n{chart_analysis}\n\n"
                "CSV数据分析：\n{csv_analysis}\n\n"
                "请生成一份完整的数据分析报告。"
            ).format(
                chart_analysis=chart_analysis or "未提供图表",
                csv_analysis=csv_analysis or "未提供CSV数据"
            )
            
            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=user_content)
            ]
            
            response = self.llm.invoke(messages)
            report = response.content
            
            logger.info("Data analysis report generated")
            
            return {
                "report": report,
                "chart_analysis": chart_analysis,
                "csv_analysis": csv_analysis
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            return {
                "error": f"生成报告时出错: {str(e)}",
                "report": ""
            }
    
    def analyze_chart(self, image_path: str = None, image_base64: str = None) -> Dict[str, Any]:
        """分析图表的便捷方法"""
        task_info = {
            "metadata": {
                "task_type": "chart_analysis",
                "image_path": image_path,
                "image_base64": image_base64
            }
        }
        return self._handle_chart_analysis_task(task_info)
    
    def analyze_csv(self, csv_path: str = None, csv_data: str = None) -> Dict[str, Any]:
        """分析CSV的便捷方法"""
        task_info = {
            "metadata": {
                "task_type": "csv_analysis",
                "csv_path": csv_path,
                "csv_data": csv_data
            }
        }
        return self._handle_csv_analysis_task(task_info)
    
    def generate_report(self, image_path: str = None, image_base64: str = None,
                       csv_path: str = None, csv_data: str = None,
                       user_requirements: str = None) -> Dict[str, Any]:
        """生成综合分析报告的便捷方法"""
        task_info = {
            "metadata": {
                "task_type": "data_analysis",
                "image_path": image_path,
                "image_base64": image_base64,
                "csv_path": csv_path,
                "csv_data": csv_data,
                "user_requirements": user_requirements
            }
        }
        return self._handle_data_analysis_task(task_info)

