# 实验三：Pandas 数据分析入门
print("--- 实验三：Pandas 数据分析入门 ---")
import pandas as pd

# 2. 创建 DataFrame
print("\n### 2. 创建 DataFrame ###")
data_dict = {
    '姓名': ['小明', '小红', '小刚', '小丽'],
    '年龄': [18, 19, 17, 18],
    '城市': ['北京', '上海', '广州', '北京'],
    '分数': [95, 88, 92, 98]
}
df = pd.DataFrame(data_dict)
print("创建的DataFrame:")
print(df)

# 3. 读取外部数据
print("\n### 3. 读取外部数据 ###")
try:
    # 确保 students.csv 文件与此脚本在同一目录下
    csv_df = pd.read_csv('week1/students.csv')
    print("\n从CSV文件读取的DataFrame:")
    print(csv_df)

    # 4. 数据查看与基本信息
    print("\n### 4. 数据查看与基本信息 ###")
    print("数据前3行:\n", csv_df.head(3))
    print("\n数据基本信息:")
    csv_df.info()
    print("\n数值数据统计描述:\n", csv_df.describe())

    # 5. 数据选择与筛选
    print("\n### 5. 数据选择与筛选 ###")
    # 选择单列
    names = csv_df['name']
    print("\n所有姓名:\n", names)

    # 选择多列
    name_and_score = csv_df[['name', 'score']]
    print("\n姓名和分数:\n", name_and_score)

    # 条件筛选
    high_scores = csv_df[csv_df['score'] > 90]
    print("\n分数高于90分的学生:\n", high_scores)
    beijing_students = csv_df[csv_df['city'] == '北京']
    print("\n所有来自北京的学生:\n", beijing_students)

    # 6. 练习任务
    print("\n### 6. Pandas 练习任务 ###")
    # 1. 筛选年龄
    young_students = csv_df[csv_df['age'] < 23]
    print("年龄小于23岁的学生:\n", young_students)

    # 2. 计算平均值
    avg_age = csv_df['age'].mean()
    avg_score = csv_df['score'].mean()
    print(f"\n平均年龄: {avg_age:.2f}")
    print(f"平均分数: {avg_score:.2f}")

    # 3. 找出最高分
    top_student = csv_df.sort_values(by='score', ascending=False).iloc[0]
    print(f"\n分数最高的学生是: {top_student['name']}")

except FileNotFoundError:
    print("\n错误: 'students.csv' 文件未找到。请确保该文件与脚本在同一目录下。")
