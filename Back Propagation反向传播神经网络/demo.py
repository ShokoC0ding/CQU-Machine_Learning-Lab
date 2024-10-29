import numpy as np

from BackPropagationNN import NeuralNetwork

from sklearn import datasets
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import metrics



def targetToVector(x):
    '''
        将类别标签转换为独热编码
    :param x:
    :return:
    '''

    # Vector
    a = np.zeros([len(x), 10])
    for i in range(0, len(x)):
        a[i, x[i]] = 1
    return a


def myTarget2Vector(x):
    '''

    :param x: 类别标签
    :return: 类别标签的独热编码
    '''

    a = np.zeros([len(x),3])
    for i in range(0,len(x)):
        a[i,x[i]] = 1
    return a


if __name__ == '__main__':

    # digits = datasets.load_digits()
    # X = preprocessing.scale(digits.data.astype(float))
    # y = targetToVector(digits.target)

    '''
        加载 鸢尾花数据集
        X:
    '''
    iris = datasets.load_iris()
    ''' preprocessing.scale 数据预处理
        .astype(float) 数据类型转换, 转换为float形式'''
    X = preprocessing.scale(iris.data.astype(float))
    y = myTarget2Vector(iris.target)

    # Cross valitation
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=10)

    # Neural Network initialization
    NN = NeuralNetwork(4, 60, 3, output_act='softmax')
    NN.fit(X_train, y_train, epochs=100, learning_rate=.1, learning_rate_decay=.01, verbose=1)

    # NN predictions0
    y_predicted = NN.predict(X_test)

    # Metrics
    y_predicted = np.argmax(y_predicted, axis=1).astype(int)
    y_test = np.argmax(y_test, axis=1).astype(int)
    print(y_predicted)
    print(y_test)

    print("\nClassification report for classifier:\n\n%s\n"
          % (metrics.classification_report(y_test, y_predicted)))
    print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_test, y_predicted))
