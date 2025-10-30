# 实验二：NumPy 数值计算实践
print("--- 实验二：NumPy 数值计算实践 ---")
import numpy as np

# 2. 创建数组 (ndarray)
print("\n### 2. 创建数组 ###")
my_list = [1, 2, 3, 4, 5]
np_array = np.array(my_list)
print(f"从列表创建的数组: {np_array}")

zeros_matrix = np.zeros((2, 3))  # 2行3列
print("2x3的全零矩阵:\n", zeros_matrix)

seq_array = np.arange(0, 10, 2)  # 从0到10(不含)，步长为2
print(f"序列数组: {seq_array}")

# 3. 数组索引与切片
print("\n### 3. 数组索引与切片 ###")
data = np.arange(10, 20)
print(f"原始数据: {data}")
print(f"第一个元素: {data[0]}")
print(f"最后三个元素: {data[-3:]}")
print(f"切片(索引1到4): {data[1:5]}")

# 4. 向量化运算与广播机制
print("\n### 4. 向量化运算与广播机制 ###")
a = np.array([10, 20, 30, 40])
b = np.array([1, 2, 3, 4])

# 元素级运算
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")

# 广播机制 (scaler + array)
print(f"a + 100 = {a + 100}")
print(f"a * 2 = {a * 2}")

# 5. 练习任务
print("\n### 5. NumPy 练习任务 ###")
# 1. 创建随机矩阵
task_matrix = np.random.randint(1, 101, size=(5, 5))
print("随机矩阵:\n", task_matrix)

# 2. 计算统计值
max_val = task_matrix.max()
min_val = task_matrix.min()
row_means = task_matrix.mean(axis=1)
col_sums = task_matrix.sum(axis=0)

print(f"最大值: {max_val}")
print(f"最小值: {min_val}")
print(f"每行平均值: {row_means}")
print(f"每列总和: {col_sums}")
