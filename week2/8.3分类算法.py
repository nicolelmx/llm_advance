#导入库
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X,y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=42,test_size=0.3,stratify=y)

#模型初始化
log_reg = LogisticRegression(max_iter = 200)
knn = KNeighborsClassifier(n_neighbors=3)
svm = SVC(kernel='linear')
tree = DecisionTreeClassifier(random_state=42)

#创建模型字典
models = {
    "logistic regression": log_reg,
    "K-Nearest Neighbors": knn,
    "Support Vector Machine" :svm,
    "Decision Tree Classifier": tree,
}

#用for循环训练模型
for name,model in models.items():
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test,y_pred)
    print(f'{name} 的准确率是：{acc:.4f}')

    # new_flower_data = [[5.1, 3.5, 3.2, 0.3]]  # 适应sklearn算法要求
    # predicted_flower_data = model.predict(new_flower_data)
    # class_name = iris.target_names[predicted_flower_data[0]]
    # print(f'新样本{new_flower_data}被决策树预测为{class_name}')

new_flower_data = [[5.1,3.5,3.2,0.3]] #适应sklearn算法要求
predicted_flower_data = tree.predict(new_flower_data)
print(predicted_flower_data)
class_name = iris.target_names[predicted_flower_data[0]]
print(f'新样本{new_flower_data}被决策树预测为{class_name}')
