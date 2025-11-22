"""
报告导出工具 - 支持多种格式的报告导出
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class ReportExporter:
    """报告导出器"""
    
    def __init__(self, export_dir: str = "./exports"):
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def export_markdown(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """导出为Markdown格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.md"
        
        filepath = self.export_dir / filename
        
        # 构建Markdown内容
        md_content = f"""# 数据分析报告

生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
        
        # 主报告
        if "report" in report_data and report_data["report"]:
            md_content += f"## 分析报告\n\n{report_data['report']}\n\n"
        elif "analysis" in report_data and report_data["analysis"]:
            md_content += f"## 分析结果\n\n{report_data['analysis']}\n\n"
        
        # 图表分析
        if "chart_analysis" in report_data and report_data["chart_analysis"]:
            md_content += f"## 图表分析\n\n{report_data['chart_analysis']}\n\n"
        
        # CSV数据分析
        if "csv_analysis" in report_data and report_data["csv_analysis"]:
            md_content += f"## 数据分析\n\n{report_data['csv_analysis']}\n\n"
        
        # 数据质量评估
        if "quality_report" in report_data and report_data["quality_report"]:
            quality = report_data["quality_report"]
            md_content += "## 数据质量评估\n\n"
            if isinstance(quality, dict):
                overall_score = quality.get("overall_score")
                if overall_score is not None:
                    md_content += f"**总体质量分数**: {overall_score:.1f}/100\n\n"
                
                dimensions = quality.get("dimensions", {})
                if dimensions:
                    md_content += "### 各维度评分\n\n"
                    for key, value in dimensions.items():
                        if isinstance(value, dict):
                            score = value.get("score", "N/A")
                            status = value.get("status", "N/A")
                            md_content += f"- **{key}**: {score if isinstance(score, str) else f'{score:.1f}'} - {status}\n"
            md_content += "\n"
        
        # 异常值检测
        if "anomalies" in report_data and report_data["anomalies"]:
            anomalies = report_data["anomalies"]
            if isinstance(anomalies, dict):
                total = anomalies.get("total_anomalies", 0)
                if total > 0:
                    md_content += f"## 异常值检测\n\n"
                    md_content += f"检测到 **{total}** 个异常值\n\n"
                    columns = anomalies.get("columns_with_anomalies", [])
                    if columns:
                        md_content += f"涉及列: {', '.join(columns)}\n\n"
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Report exported to {filepath}")
        return str(filepath)
    
    def export_json(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """导出为JSON格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.json"
        
        filepath = self.export_dir / filename
        
        export_data = {
            "generated_at": datetime.now().isoformat(),
            "report": report_data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Report exported to {filepath}")
        return str(filepath)
    
    def export_html(self, report_data: Dict[str, Any], filename: Optional[str] = None) -> str:
        """导出为HTML格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{timestamp}.html"
        
        filepath = self.export_dir / filename
        
        html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>数据分析报告</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }}
        h1 {{
            color: #667eea;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #764ba2;
            margin-top: 30px;
        }}
        .meta {{
            color: #666;
            font-size: 0.9em;
            margin-bottom: 30px;
        }}
        pre {{
            background: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <h1>📊 数据分析报告</h1>
    <div class="meta">生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
"""
        
        # 主报告
        if "report" in report_data and report_data["report"]:
            report_html = report_data["report"].replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<h2>分析报告</h2><div style='white-space: pre-wrap;'>{report_html}</div>"
        elif "analysis" in report_data and report_data["analysis"]:
            analysis_html = report_data["analysis"].replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<h2>分析结果</h2><div style='white-space: pre-wrap;'>{analysis_html}</div>"
        
        # 图表分析
        if "chart_analysis" in report_data and report_data["chart_analysis"]:
            chart_html = report_data["chart_analysis"].replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<h2>图表分析</h2><div style='white-space: pre-wrap;'>{chart_html}</div>"
        
        # CSV数据分析
        if "csv_analysis" in report_data and report_data["csv_analysis"]:
            csv_html = report_data["csv_analysis"].replace('\n', '<br>').replace('<', '&lt;').replace('>', '&gt;')
            html_content += f"<h2>数据分析</h2><div style='white-space: pre-wrap;'>{csv_html}</div>"
        
        # 数据质量评估
        if "quality_report" in report_data and report_data["quality_report"]:
            quality = report_data["quality_report"]
            html_content += "<h2>数据质量评估</h2><div>"
            if isinstance(quality, dict):
                overall_score = quality.get("overall_score")
                if overall_score is not None:
                    score_color = "#28a745" if overall_score >= 80 else "#ffc107" if overall_score >= 60 else "#dc3545"
                    html_content += f"<p><strong>总体质量分数</strong>: <span style='font-size: 1.5em; color: {score_color}'>{overall_score:.1f}</span>/100</p>"
                
                dimensions = quality.get("dimensions", {})
                if dimensions:
                    html_content += "<h3>各维度评分</h3><ul>"
                    for key, value in dimensions.items():
                        if isinstance(value, dict):
                            score = value.get("score", "N/A")
                            status = value.get("status", "N/A")
                            score_text = score if isinstance(score, str) else f"{score:.1f}"
                            html_content += f"<li><strong>{key}</strong>: {score_text} - {status}</li>"
                    html_content += "</ul>"
            html_content += "</div>"
        
        # 异常值检测
        if "anomalies" in report_data and report_data["anomalies"]:
            anomalies = report_data["anomalies"]
            if isinstance(anomalies, dict):
                total = anomalies.get("total_anomalies", 0)
                if total > 0:
                    html_content += f"<h2>异常值检测</h2><div style='background: #fff3cd; padding: 15px; border-radius: 5px;'>"
                    html_content += f"<p>检测到 <strong>{total}</strong> 个异常值</p>"
                    columns = anomalies.get("columns_with_anomalies", [])
                    if columns:
                        html_content += f"<p>涉及列: {', '.join(columns)}</p>"
                    html_content += "</div>"
        
        html_content += """
</body>
</html>
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Report exported to {filepath}")
        return str(filepath)

