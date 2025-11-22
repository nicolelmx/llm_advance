"""
图表生成工具 - 根据数据分析结果生成可视化图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
import base64
import io
from typing import Dict, List, Any, Optional
import logging

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

logger = logging.getLogger(__name__)

class ChartGenerator:
    """图表生成器"""
    
    @staticmethod
    def generate_summary_charts(df: pd.DataFrame, output_dir: str = "./charts") -> Dict[str, str]:
        """生成数据摘要图表"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        charts = {}
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) == 0:
            return charts
        
        # 1. 数值列分布图
        if len(numeric_cols) > 0:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('数据概览', fontsize=16, fontweight='bold')
            
            # 选择前4个数值列
            cols_to_plot = numeric_cols[:4]
            
            for idx, col in enumerate(cols_to_plot):
                ax = axes[idx // 2, idx % 2]
                df[col].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
                ax.set_title(f'{col} 分布', fontsize=12)
                ax.set_xlabel(col)
                ax.set_ylabel('频数')
            
            # 隐藏多余的子图
            for idx in range(len(cols_to_plot), 4):
                axes[idx // 2, idx % 2].axis('off')
            
            plt.tight_layout()
            chart_path = os.path.join(output_dir, 'data_distribution.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            charts['distribution'] = chart_path
        
        # 2. 相关性热力图
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            corr_matrix = df[numeric_cols].corr()
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
            ax.set_title('变量相关性热力图', fontsize=14, fontweight='bold')
            plt.tight_layout()
            chart_path = os.path.join(output_dir, 'correlation_heatmap.png')
            plt.savefig(chart_path, dpi=150, bbox_inches='tight')
            plt.close()
            charts['correlation'] = chart_path
        
        return charts
    
    @staticmethod
    def chart_to_base64(chart_path: str) -> str:
        """将图表转换为base64编码"""
        try:
            with open(chart_path, 'rb') as f:
                image_data = f.read()
            return base64.b64encode(image_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Error converting chart to base64: {str(e)}")
            return ""
    
    @staticmethod
    def generate_trend_chart(df: pd.DataFrame, date_col: str, value_col: str, 
                            output_path: Optional[str] = None) -> str:
        """生成趋势图"""
        try:
            # 确保日期列是datetime类型
            if df[date_col].dtype != 'datetime64[ns]':
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            
            # 按日期排序
            df_sorted = df.sort_values(date_col)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df_sorted[date_col], df_sorted[value_col], marker='o', linewidth=2, markersize=4)
            ax.set_title(f'{value_col} 趋势图', fontsize=14, fontweight='bold')
            ax.set_xlabel('日期')
            ax.set_ylabel(value_col)
            ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                # 保存到内存
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            
            plt.close()
            return output_path
        except Exception as e:
            logger.error(f"Error generating trend chart: {str(e)}")
            return ""
    
    @staticmethod
    def generate_comparison_chart(df: pd.DataFrame, category_col: str, value_col: str,
                                 chart_type: str = 'bar', output_path: Optional[str] = None) -> str:
        """生成对比图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if chart_type == 'bar':
                df.groupby(category_col)[value_col].mean().plot(kind='bar', ax=ax, color='coral')
                ax.set_title(f'{value_col} 按 {category_col} 对比', fontsize=14, fontweight='bold')
            elif chart_type == 'pie':
                df.groupby(category_col)[value_col].sum().plot(kind='pie', ax=ax, autopct='%1.1f%%')
                ax.set_title(f'{value_col} 分布', fontsize=14, fontweight='bold')
                ax.set_ylabel('')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
            else:
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                return base64.b64encode(buffer.read()).decode('utf-8')
            
            plt.close()
            return output_path
        except Exception as e:
            logger.error(f"Error generating comparison chart: {str(e)}")
            return ""

