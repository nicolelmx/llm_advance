import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def setup_pycharm_plotting():
    """设置PyCharm专用的绘图配置"""

    # 尝试不同的后端
    backends_to_try = [
        'TkAgg',  # 通常最稳定
        'Qt5Agg',  # 需要安装PyQt5
        'MacOSX',  # macOS原生
        'WebAgg',  # 在浏览器中显示
    ]

    for backend in backends_to_try:
        try:
            matplotlib.use(backend)
            print(f"成功设置后端: {backend}")
            return True
        except:
            continue

    print("无法设置任何后端，使用默认设置")
    return False


# 设置后端
setup_pycharm_plotting()

# 现在创建图表
plt.figure(figsize=(10, 6))
data = np.random.randn(1000)
plt.hist(data, bins=30, alpha=0.7, color='blue')
plt.title('随机数据分布')
plt.xlabel('数值')
plt.ylabel('频率')

plt.tight_layout()
plt.show()