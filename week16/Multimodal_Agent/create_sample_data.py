"""
创建示例数据文件用于测试
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

# 创建示例CSV数据
def create_sample_csv():
    """创建示例CSV文件"""
    data = {
        '日期': pd.date_range('2024-01-01', periods=30, freq='D'),
        '销售额': [10000 + i*500 + (i%7)*200 for i in range(30)],
        '订单数': [50 + i*2 + (i%5)*5 for i in range(30)],
        '用户数': [100 + i*3 + (i%3)*10 for i in range(30)]
    }
    df = pd.DataFrame(data)
    df.to_csv('sample_data.csv', index=False, encoding='utf-8-sig')
    print("✓ 已创建 sample_data.csv")

# 创建示例图表
def create_sample_chart():
    """创建示例图表"""
    data = {
        '月份': ['1月', '2月', '3月', '4月', '5月', '6月'],
        '销售额': [10000, 12000, 15000, 11000, 13000, 16000],
        '订单数': [50, 60, 75, 55, 65, 80]
    }
    df = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 柱状图
    ax1.bar(df['月份'], df['销售额'], color='skyblue')
    ax1.set_title('月度销售额', fontsize=14, fontproperties='SimHei')
    ax1.set_xlabel('月份', fontproperties='SimHei')
    ax1.set_ylabel('销售额（元）', fontproperties='SimHei')
    
    # 折线图
    ax2.plot(df['月份'], df['订单数'], marker='o', color='coral', linewidth=2)
    ax2.set_title('月度订单数', fontsize=14, fontproperties='SimHei')
    ax2.set_xlabel('月份', fontproperties='SimHei')
    ax2.set_ylabel('订单数', fontproperties='SimHei')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sample_chart.png', dpi=150, bbox_inches='tight')
    print("✓ 已创建 sample_chart.png")

if __name__ == "__main__":
    try:
        create_sample_csv()
        create_sample_chart()
        print("\n示例文件已创建完成！")
        print("可以使用这些文件测试系统功能。")
    except Exception as e:
        print(f"创建示例文件时出错: {str(e)}")
        print("请确保已安装 pandas 和 matplotlib")

