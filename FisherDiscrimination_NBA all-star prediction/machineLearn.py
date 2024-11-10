import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve,roc_auc_score
from matplotlib.colors import ListedColormap


def calculate_metrics(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1-score for predictions.
    """
    # 调库计算 准确率、精确率、召回率、F1分数
    accuracy = accuracy_score(y_true,y_pred)
    precision = precision_score(y_true,y_pred,average='macro')
    recall = recall_score(y_true,y_pred,average='macro')
    f1 = f1_score(y_true,y_pred,average='macro')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def manual_metrics_calculation(y_true, y_pred):
    """
    Manually calculate accuracy, precision, recall, and F1-score for predictions.

    Args:
    y_true (array): Actual true labels.
    y_pred (array): Predicted labels.

    Returns:
    dict: Dictionary containing accuracy, precision, recall, and F1-score.
    """
    accuracy = 0.
    precision = 0.
    recall = 0.
    f1 = 0.

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def set_predict(x,y,y_predict):

    plt.figure(figsize=(10, 4))
    # 绘制真实数据点
    plt.plot(x[y == 2, 0], x[y == 2, 1], 'go', label='Predicted')
    plt.plot(x[y == 1, 0], x[y == 1, 1], 'bo')
    plt.plot(x[y == 0, 0], x[y == 0, 1], 'yo')
    # 绘制预测数据点
    plt.plot(x[y_predict == 2, 0], x[y_predict == 2, 1], 'go',marker='x',label='Actual')
    plt.plot(x[y_predict == 1, 0], x[y_predict == 1, 1], 'bo', marker='x')
    plt.plot(x[y_predict == 0, 0], x[y_predict == 0, 1], 'yo', marker='x')

    plt.legend(loc='upper left', fontsize=14)
    plt.title('Set Prediction')
    plt.xlabel('Attribute 1')
    plt.ylabel('Attribute 2')
    plt.show()

    return -1


if __name__ == "__main__":
    # 读取数据
    iris_data = datasets.load_iris()
    attribute1 = 0
    attribute2 = 1
    category = ['Sepal length','Sepal width','Petal length','Petal width']
    # 可以通过数字序号 0、1、2、3 选择四种属性 'Sepal length','Sepal width','Petal length','Petal width'
    X = iris_data['data'][:, (attribute1, attribute2)]
    y = iris_data['target']
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #  在训练集上进行训练
    lr = LogisticRegression(multi_class='multinomial', solver='lbfgs').fit(X_train, y_train)

    #  在训练集上预测，并绘制预测散点图
    y_proba_train = lr.predict_proba(X_train)
    y_predict_train = lr.predict(X_train)
    set_predict(X_train,y_train,y_predict_train)
    #  在测试集上进行预测，并绘制预测散点图
    y_proba = lr.predict_proba(X_test)
    y_predict = lr.predict(X_test)
    set_predict(X_test,y_test,y_predict)

    # 设置坐标轴起终点
    x_start = iris_data['data'][:,attribute1].min()-1
    x_end = iris_data['data'][:,attribute1].max()+1
    y_start = iris_data['data'][:,attribute2].min()-1
    y_end = iris_data['data'][:,attribute2].max()+1
    # 绘制网格
    x0, x1 = np.meshgrid\
            (np.linspace(x_start,x_end,500).reshape(-1, 1),
             np.linspace(y_start,y_end,200).reshape(-1, 1)
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    # 在整个网格上进行预测
    y_proba_grid = lr.predict_proba(X_new)
    y_predict_grid = lr.predict(X_new)
    zz1 = y_proba_grid[:, 1].reshape(x0.shape)
    zz = y_predict_grid.reshape(x0.shape)

    #  绘制决策边界

    plt.figure(figsize=(10, 4))
    plt.plot(X[y == 2, 0], X[y == 2, 1], 'g^', label='Iris-Virginica')
    plt.plot(X[y == 1, 0], X[y == 1, 1], 'bs', label='Iris-Versicolor')
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'yo', label='Iris-Setosa')

    # 绘制填充等高线图
    custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])  # 一个颜色映射对象
    plt.contourf(x0, x1, zz, cmap=custom_cmap,alpha=0.3) # alpha设置透明度

    plt.title('Decision Boundary')
    plt.xlabel(category[attribute1], fontsize=14)
    plt.ylabel(category[attribute2], fontsize=14)
    plt.legend(loc='center left', fontsize=14)
    plt.axis([x_start, x_end, y_start, y_end])
    plt.show()

    print(calculate_metrics(y_test,y_predict))

