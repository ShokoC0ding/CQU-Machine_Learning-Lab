import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn import metrics


class NeuralNetwork(object):

    '''
            属性:
        activation
        activation_prime
        output_act
        output_act_prime
        wi
        wo
        updatei
        updateo
    '''

    def __init__(self, inputs, hidden, outputs, activation='tanh', output_act='softmax'):

        '''Hidden layer activation function 隐藏层激活函数'''
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime
        elif activation == 'linear':
            self.activation = linear
            self.activation_prime = linear_prime

        '''Output layer activation function 输出层激活函数'''
        if output_act == 'sigmoid':
            self.output_act = sigmoid
            self.output_act_prime = sigmoid_prime
        elif output_act == 'tanh':
            self.output_act = tanh
            self.output_act_prime = tanh_prime
        elif output_act == 'linear':
            self.output_act = linear
            self.output_act_prime = linear_prime
        elif output_act == 'softmax':
            self.output_act = softmax
            self.output_act_prime = softmax_prime

        '''Weights initializarion 输入层到隐藏层的权重初始化'''
        self.wi = np.random.randn(inputs, hidden) / np.sqrt(inputs)
        self.wo = np.random.randn(hidden + 1, outputs) / np.sqrt(hidden)

        '''Weights updates initialization  隐藏层到输出层的权重初始化'''
        self.updatei = 0
        self.updateo = 0

    def feedforward(self, X):
        """
        :param X: 输入矩阵 X
        :return: 输出标签 y
        """
        # Hidden layer activation
        ah = self.activation(np.dot(X, self.wi))

        # Adding bias to the hidden layer results
        ''' np.concatenate((array_1,array_2...array_n),axis=0,out=None)
            连接数组，必须是相同形状
            '''
        ah = np.concatenate((np.ones(1).T, np.array(ah)))

        # Outputs
        y = self.output_act(np.dot(ah, self.wo))

        # Return the results
        return y

    def fit(self, X, y, epochs, learning_rate=0.2, learning_rate_decay=0, momentum=0, verbose=0):
        """

        :param X:
        :param y:
        :param epochs: 训练轮次
        :param learning_rate: 学习率衰减
        :param learning_rate_decay:
        :param momentum: 动量
        :param verbose:
        :return:

        verbose是日志显示，
        verbose=0,不输出日志信息; verbose=1, 带进度条地输出日志信息; verbose=2, 为每个epoch输出一行记录，没有进度条
        """

        y_argmax = np.argmax(y, axis=1).astype(int)
        precision_list=[]
        accuracy_list=[]
        loss_list=[]
        recall_list=[]

        # Timer start
        startTime = time.time()

        # Epochs loop
        for k in range(epochs):

            # Dataset loop
            for i in range(X.shape[0]):
                # Hidden layer activation
                ah = self.activation(np.dot(X[i], self.wi))

                # Adding bias to the hidden layer
                ah = np.concatenate((np.ones(1).T, np.array(ah)))

                # Output activation
                ao = self.output_act(np.dot(ah, self.wo))

                # Deltas
                deltao = np.multiply(self.output_act_prime(ao), y[i] - ao)
                deltai = np.multiply(self.activation_prime(ah), np.dot(self.wo, deltao))

                # Weights update with momentum
                self.updateo = momentum * self.updateo + np.multiply(learning_rate, np.outer(ah, deltao))
                self.updatei = momentum * self.updatei + np.multiply(learning_rate, np.outer(X[i], deltai[1:]))

                # Weights update
                self.wo += self.updateo
                self.wi += self.updatei

            # Print training status
            if verbose == 1:
                print('EPOCH: {0:4d}/{1:4d}\t\tLearning rate: {2:4f}\t\tElapse time [seconds]: {3:5f}'.format(k, epochs,
                                                                                                        learning_rate,
                                                                                                        time.time() - startTime))
            # Learning rate update
            learning_rate = learning_rate * (1 - learning_rate_decay)

            y_in_epoch = np.argmax(self.predict(X), axis=1).astype(int)

            # Precision
            precision = metrics.precision_score(y_argmax,y_in_epoch,average='macro')
            precision_list.append(precision)
            # Accuracy
            accuracy = metrics.accuracy_score(y_argmax,y_in_epoch)
            accuracy_list.append(accuracy)
            # Loss
            loss = metrics.log_loss(y,self.predict(X))
            loss_list.append(loss)
            # Recall  对数损失
            recall = metrics.recall_score(y_argmax,y_in_epoch,average='macro')
            recall_list.append(recall)

        fig,axes=plt.subplots(nrows=2,ncols=2)
        fig.subplots_adjust(wspace=0.5,hspace=0.3,left=0.125,right=0.9,top=0.9,bottom=0.1)
        # 设置子图
        axes[0,0].set(title='Precision Over Epochs')
        axes[0,1].set(title='Accuracy Over Epochs')
        axes[1,0].set(title='Loss Over Epochs')
        axes[1,1].set(title='Recall Over Epochs')

        # 设置横纵坐标
        x = np.linspace(0, epochs,epochs)
        precision_coord = precision_list
        accuracy_coord = accuracy_list
        loss_coord = loss_list
        recall_coord = recall_list

        # 绘图
        axes[0,0].plot(x,precision_coord)
        axes[0,1].plot(x,accuracy_coord)
        axes[1,0].plot(x,loss_coord)
        axes[1,1].plot(x,recall_coord)
        plt.show()


    def predict(self, X):

        # Allocate memory for the outputs
        y = np.zeros([X.shape[0], self.wo.shape[1]])

        # Loop the inputs
        for i in range(0, X.shape[0]):
            y[i] = self.feedforward(X[i])
        # Return the results

        return y


# Activation functions
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_prime(x):
    return sigmoid(x) * (1.0 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1.0 - x ** 2


def softmax(x):
    return (np.exp(np.array(x)) / np.sum(np.exp(np.array(x))))


def softmax_prime(x):
    return softmax(x) * (1.0 - softmax(x))


def linear(x):
    return x


def linear_prime(x):
    return 1