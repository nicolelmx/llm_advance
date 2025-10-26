# 创建一个不平衡的数据集，数据集往往存在不平衡的情况，比如正负样本的比例为1:100，这样会导致模型倾向于预测负样本，从而导致模型性能下降。
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 创建一个包含1000个样本的数据集，其中95%是类别0，5%是类别1
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, # 只有五个主要特征
                           n_redundant=0, n_classes=2, n_clusters_per_class=1,
                           weights=[0.95, 0.05], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 在这个不平衡数据集上训练一个逻辑回归模型
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 准确率的“陷阱”
acc = accuracy_score(y_test, y_pred)
print(f"不平衡数据集上的准确率: {acc:.4f}")

# 使用混淆矩阵和分类报告来查看模型的性能
# 混淆矩阵
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# 分类报告
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=["类别 0 (多数类)", "类别 1 (少数类)"]))


