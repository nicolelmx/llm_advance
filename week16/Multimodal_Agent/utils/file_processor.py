"""
文件处理工具 - 支持多种文件格式的读取和解析
"""

import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import io

logger = logging.getLogger(__name__)

class FileProcessor:
    """文件处理器 - 支持多种格式"""
    
    @staticmethod
    def detect_file_type(filename: str) -> str:
        """检测文件类型"""
        ext = Path(filename).suffix.lower()
        type_mapping = {
            '.csv': 'csv',
            '.xlsx': 'excel',
            '.xls': 'excel',
            '.json': 'json',
            '.txt': 'text',
            '.tsv': 'tsv'
        }
        return type_mapping.get(ext, 'unknown')
    
    @staticmethod
    def read_csv(file_path: str = None, file_data: str = None, encoding: str = 'utf-8') -> pd.DataFrame:
        """读取CSV文件"""
        try:
            if file_data:
                return pd.read_csv(io.StringIO(file_data), encoding=encoding)
            else:
                return pd.read_csv(file_path, encoding=encoding)
        except UnicodeDecodeError:
            # 尝试其他编码
            try:
                if file_data:
                    return pd.read_csv(io.StringIO(file_data), encoding='gbk')
                else:
                    return pd.read_csv(file_path, encoding='gbk')
            except Exception as e:
                logger.error(f"Error reading CSV: {str(e)}")
                raise
    
    @staticmethod
    def read_excel(file_path: str = None, file_data: bytes = None, sheet_name: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """读取Excel文件，返回多个sheet的字典"""
        try:
            if file_data:
                excel_file = io.BytesIO(file_data)
                excel_data = pd.read_excel(excel_file, sheet_name=None if sheet_name is None else sheet_name)
            else:
                excel_data = pd.read_excel(file_path, sheet_name=None if sheet_name is None else sheet_name)
            
            # 如果只有一个sheet，返回字典格式
            if isinstance(excel_data, pd.DataFrame):
                return {"Sheet1": excel_data}
            return excel_data
        except Exception as e:
            logger.error(f"Error reading Excel: {str(e)}")
            raise
    
    @staticmethod
    def read_json(file_path: str = None, file_data: str = None) -> Any:
        """读取JSON文件"""
        try:
            if file_data:
                return json.loads(file_data)
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error reading JSON: {str(e)}")
            raise
    
    @staticmethod
    def read_file(file_path: str = None, file_data: bytes = None, filename: str = "data") -> Dict[str, Any]:
        """通用文件读取方法"""
        file_type = FileProcessor.detect_file_type(filename)
        
        result = {
            "file_type": file_type,
            "filename": filename,
            "data": None,
            "sheets": None,
            "error": None
        }
        
        try:
            if file_type == 'csv' or file_type == 'tsv':
                if file_data:
                    df = FileProcessor.read_csv(file_data=file_data.decode('utf-8'))
                else:
                    df = FileProcessor.read_csv(file_path=file_path)
                result["data"] = df
                result["sheets"] = {"data": df}
            
            elif file_type == 'excel':
                if file_data:
                    sheets = FileProcessor.read_excel(file_data=file_data)
                else:
                    sheets = FileProcessor.read_excel(file_path=file_path)
                result["sheets"] = sheets
                # 默认使用第一个sheet作为主数据
                result["data"] = list(sheets.values())[0] if sheets else None
            
            elif file_type == 'json':
                if file_data:
                    json_data = FileProcessor.read_json(file_data=file_data.decode('utf-8'))
                else:
                    json_data = FileProcessor.read_json(file_path=file_path)
                
                # 尝试将JSON转换为DataFrame
                if isinstance(json_data, list) and len(json_data) > 0:
                    try:
                        df = pd.DataFrame(json_data)
                        result["data"] = df
                        result["sheets"] = {"data": df}
                    except:
                        result["data"] = json_data
                elif isinstance(json_data, dict):
                    # 如果是字典，尝试转换为DataFrame
                    try:
                        df = pd.DataFrame([json_data])
                        result["data"] = df
                        result["sheets"] = {"data": df}
                    except:
                        result["data"] = json_data
                else:
                    result["data"] = json_data
            
            else:
                result["error"] = f"Unsupported file type: {file_type}"
        
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error processing file {filename}: {str(e)}")
        
        return result
    
    @staticmethod
    def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """获取数据摘要信息"""
        if df is None or df.empty:
            return {"error": "DataFrame is empty"}
        
        # 导入json_utils用于序列化
        from utils.json_utils import convert_to_serializable
        
        summary = {
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.astype(str).to_dict().items()},
            "null_counts": {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            "null_percentages": {str(k): float(v) for k, v in (df.isnull().sum() / len(df) * 100).to_dict().items()},
            "memory_usage": int(df.memory_usage(deep=True).sum()),
            "sample_data": convert_to_serializable(df.head(10).to_dict('records'))
        }
        
        # 数值列统计
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_stats"] = convert_to_serializable(df[numeric_cols].describe().to_dict())
        
        # 分类列统计
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary["categorical_info"] = {}
            for col in categorical_cols[:5]:  # 限制前5个分类列
                summary["categorical_info"][col] = {
                    "unique_count": int(df[col].nunique()),
                    "top_values": {str(k): int(v) for k, v in df[col].value_counts().head(5).to_dict().items()}
                }
        
        return summary

