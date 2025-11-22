"""
数据质量评估工具 - 评估数据质量、检测异常值
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    @staticmethod
    def assess_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """评估数据质量"""
        if df is None or df.empty:
            return {"error": "DataFrame is empty"}
        
        quality_report = {
            "overall_score": 0,
            "dimensions": {}
        }
        
        # 1. 完整性评估
        completeness = DataQualityAnalyzer._assess_completeness(df)
        quality_report["dimensions"]["completeness"] = completeness
        
        # 2. 准确性评估（检查明显错误）
        accuracy = DataQualityAnalyzer._assess_accuracy(df)
        quality_report["dimensions"]["accuracy"] = accuracy
        
        # 3. 一致性评估
        consistency = DataQualityAnalyzer._assess_consistency(df)
        quality_report["dimensions"]["consistency"] = consistency
        
        # 4. 及时性评估（如果有时间列）
        timeliness = DataQualityAnalyzer._assess_timeliness(df)
        quality_report["dimensions"]["timeliness"] = timeliness
        
        # 计算总体质量分数
        scores = []
        for dim_name, dim_data in quality_report["dimensions"].items():
            if isinstance(dim_data, dict):
                score = dim_data.get("score")
                if score is not None:
                    try:
                        scores.append(float(score))
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid score in dimension {dim_name}: {score}")
        
        if scores:
            quality_report["overall_score"] = float(sum(scores) / len(scores))
        else:
            quality_report["overall_score"] = 0.0
            logger.warning("No valid scores found in quality dimensions")
        
        return quality_report
    
    @staticmethod
    def _assess_completeness(df: pd.DataFrame) -> Dict[str, Any]:
        """评估完整性"""
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        completeness_rate = (1 - null_cells / total_cells) * 100
        
        return {
            "score": float(completeness_rate),
            "total_cells": int(total_cells),
            "null_cells": int(null_cells),
            "completeness_rate": f"{completeness_rate:.2f}%",
            "status": "优秀" if completeness_rate >= 95 else "良好" if completeness_rate >= 80 else "需改进"
        }
    
    @staticmethod
    def _assess_accuracy(df: pd.DataFrame) -> Dict[str, Any]:
        """评估准确性"""
        issues = []
        
        # 检查数值列的异常值
        numeric_cols = df.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            try:
                # 跳过全为NaN的列
                if df[col].isna().all():
                    continue
                    
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                
                # 如果Q1和Q3相同，说明数据没有变化，跳过
                if pd.isna(Q1) or pd.isna(Q3) or Q1 == Q3:
                    continue
                    
                IQR = Q3 - Q1
                if IQR > 0:  # 确保IQR有效
                    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
                    if len(outliers) > 0:
                        issues.append(f"{col}: {len(outliers)}个异常值")
            except Exception as e:
                logger.warning(f"Error assessing accuracy for column {col}: {str(e)}")
                continue
        
        # 检查重复行
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"重复行: {duplicates}行")
        
        # 计算准确性分数（每个问题扣10分，最低0分）
        accuracy_score = max(0, 100 - len(issues) * 10)
        
        return {
            "score": float(accuracy_score),
            "issues": issues,
            "duplicates": int(duplicates),
            "status": "优秀" if accuracy_score >= 90 else "良好" if accuracy_score >= 70 else "需改进"
        }
    
    @staticmethod
    def _assess_consistency(df: pd.DataFrame) -> Dict[str, Any]:
        """评估一致性"""
        issues = []
        
        # 检查数据类型一致性
        for col in df.columns:
            if df[col].dtype == 'object':
                # 检查是否有混合类型
                try:
                    pd.to_numeric(df[col], errors='raise')
                except:
                    pass  # 正常，是文本类型
                else:
                    issues.append(f"{col}: 可能包含数值但被存储为文本")
        
        consistency_score = max(0, 100 - len(issues) * 15)
        
        return {
            "score": float(consistency_score),
            "issues": issues,
            "status": "优秀" if consistency_score >= 85 else "良好" if consistency_score >= 70 else "需改进"
        }
    
    @staticmethod
    def _assess_timeliness(df: pd.DataFrame) -> Dict[str, Any]:
        """评估及时性（如果有时间列）"""
        date_cols = []
        for col in df.columns:
            if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                date_cols.append(col)
        
        if not date_cols:
            return {
                "score": 100.0,
                "status": "无时间列",
                "message": "数据中未检测到时间列"
            }
        
        # 检查时间列的完整性
        timeliness_score = 100.0
        for col in date_cols:
            try:
                null_rate = df[col].isnull().sum() / len(df) * 100
                timeliness_score -= null_rate
            except Exception as e:
                logger.warning(f"Error assessing timeliness for column {col}: {str(e)}")
                continue
        
        final_score = max(0.0, timeliness_score)
        return {
            "score": float(final_score),
            "date_columns": date_cols,
            "status": "优秀" if final_score >= 90 else "良好" if final_score >= 70 else "需改进"
        }
    
    @staticmethod
    def detect_anomalies(df: pd.DataFrame, columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """检测异常值"""
        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()
        
        anomalies = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # 使用IQR方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                
                if outlier_indices:
                    anomalies[col] = {
                        "count": int(len(outlier_indices)),
                        "indices": [int(idx) for idx in outlier_indices[:10]],  # 只返回前10个，转换为int
                        "lower_bound": float(lower_bound) if not pd.isna(lower_bound) else None,
                        "upper_bound": float(upper_bound) if not pd.isna(upper_bound) else None,
                        "outlier_values": [float(v) if not pd.isna(v) else None for v in df.loc[outlier_indices[:5], col].tolist()]
                    }
        
        total_anomalies = sum(a["count"] for a in anomalies.values())
        return {
            "total_anomalies": int(total_anomalies),
            "columns_with_anomalies": list(anomalies.keys()),
            "details": anomalies
        }

