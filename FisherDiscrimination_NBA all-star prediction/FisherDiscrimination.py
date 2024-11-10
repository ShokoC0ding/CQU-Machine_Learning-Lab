from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import math
from imblearn.over_sampling import SMOTE

# 读取CSV文件
file_path = '22-23敲定_mixup.csv'
file_path2='21-22敲定.csv'
df = pd.read_csv(file_path)
df2 = pd.read_csv(file_path2)

# 将DataFrame转换为NumPy数组
data = df.to_numpy()
data2=df2.to_numpy()

# 去掉第一列和最后一列
#X = data[:, 2:-1]
X = data[:,0:-1]
y = data[:,-1]

X_21 = data2[:,0:-1]
X_21_att = data2[:, 1:-1]
y_21 = data2[:,-1]


X = X.astype(float)
y = y.astype(float)

x_train,x_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

all_star21_22=[]

#标准化处理
# from sklearn.preprocessing import StandardScaler
# ss_x=StandardScaler()
# ss_y=StandardScaler()
# #分别对训练和测试数据的特征以及目标值进行标准化处理
# x_train = ss_x.fit_transform(x_train)
# x_test = ss_x.transform(x_test)

#SMOTE处理
# sm = SMOTE(random_state=123)
# x_train, y_train = sm.fit_resample(x_train, y_train)

def get_mean_vector(target):
    '''
    求均值向量
    :param target:
    :return:
    '''
    m_target_list = [0 for i in range(x_train.shape[1])]
    count = 0
    for i in range(x_train.shape[0]):
        if y_train[i] == target:
            count = count + 1
            temp = x_train[i].tolist()
            m_target_list = [m_target_list[j] + temp[j] for j in range(x_train.shape[1])]
    m_target_list = [x / count for x in m_target_list]
    # 其实可以用类似torch的压缩维度的函数直接求和
    return m_target_list


def get_dispersion_matrix(target, mean_vector):
    '''
    求样本内离散度矩阵
    :param target:
    :param mean_vector:
    :return:
    '''
    s_target_matrix = np.zeros((x_train.shape[1], x_train.shape[1]))
    for i in range(x_train.shape[0]):
        if y_train[i] == target:
            temp = np.multiply(x_train[i] - mean_vector, (x_train[i] - mean_vector).transpose())
            s_target_matrix = s_target_matrix + temp
    return s_target_matrix


def get_sample_divergence(mean_vector1, mean_vector2):
    '''
    求样本间离散度
    :param mean_vector1:
    :param mean_vector2:
    :return:
    '''
    return np.multiply((mean_vector1 - mean_vector2), (mean_vector1 - mean_vector2).transpose())


def get_w_star(dispersion_matrix, mean_vector1, mean_vector2):
    '''
    求Fisher准则函数的w_star解
    :param dispersion_matrix:
    :param mean_vector1:
    :param mean_vector2:
    :return:
    '''
    return np.matmul(np.linalg.inv(dispersion_matrix), (mean_vector1 - mean_vector2))

def get_sample_projection(w_star, x):
    '''
    求一特征向量在w_star上的投影
    :param w_star:
    :param x:
    :return:
    '''
    return np.matmul(w_star.transpose(), x)


def get_segmentation_threshold(w_star, way_flag):
    '''
    求分割阈值
    :param w_star:
    :param way_flag:
    :return:
    '''
    if way_flag == 0:
        y0_list = []
        y1_list = []
        for i in range(x_train.shape[0]):
            if y_train[i] == 0:
                y0_list.append(get_sample_projection(w_star, x_train[i]))
            else:
                y1_list.append(get_sample_projection(w_star, x_train[i]))
        ny0 = len(y0_list)
        ny1 = len(y1_list)
        my0 = sum(y0_list) / ny0
        my1 = sum(y1_list) / ny1
        segmentation_threshold = (ny0 * my0 + ny1 * my1) / (ny0 + ny1)
        return  segmentation_threshold
    elif way_flag == 1:
        y0_list = []
        y1_list = []
        for i in range(x_train.shape[0]):
            if y_train[i] == 0:
                y0_list.append(get_sample_projection(w_star, x_train[i]))
            else:
                y1_list.append(get_sample_projection(w_star, x_train[i]))
        ny0 = len(y0_list)
        ny1 = len(y1_list)
        my0 = sum(y0_list) / ny0
        my1 = sum(y1_list) / ny1
        py0 = ny0 / (ny0 + ny1)
        py1 = ny1 / (ny0 + ny1)
        segmentation_threshold = (my0 + my1) / 2 + math.log(py0 / py1) / (ny0 - ny1 - 2)
        return segmentation_threshold
    else:
        return 0


def test_single_smaple(w_star, y0, test_sample, test_target):
    '''
    单例测试
    :param y0:
    :param x:
    :return:
    '''
    y_test = get_sample_projection(w_star, test_sample)
    predection = 1
    if y_test > y0:
        predection = 0
    print("This x_vector's target is {}, and the predection is {}".format(test_target, predection))


def test_single_smaple_check(w_star, y0, test_sample, test_target):
    '''
    单例测试（用于统计）
    :param y0:
    :param x:
    :return:
    '''
    # test_sample_att=test_sample[:,1:-1]
    y_test = get_sample_projection(w_star, test_sample)
    predection = 1
    if y_test > y0:
        predection = 0
    if test_target == predection == 1:
        return "all star"
    if test_target == predection == 0:
        return "nope"
    else:
        return 0


def test_check(w_star, y0):
    '''
    统计测试样本
    :param w_star:
    :param y0:
    :return:
    '''
    right_count = 0
    for i in range(x_test.shape[0]):
        boolean = test_single_smaple_check(w_star, y0, x_test[i], y_test[i])
        if (boolean == "all star" or boolean == "nope"):
            right_count = right_count + 1
    return x_test.shape[0], right_count, right_count / x_test.shape[0]

def new_test_check(w_star, y0):
    '''
    统计测试样本
    :param w_star:
    :param y0:
    :return:
    '''
    right_count = 0
    for i in range(X_21.shape[0]):
        boolean = test_single_smaple_check(w_star, y0, X_21_att[i], y_21[i])
        if (boolean == "all star" or boolean == "nope"):
            right_count = right_count + 1
        if (boolean == "all star"):
            all_star21_22.append(X_21[i][0])

    print(all_star21_22)
    return X_21.shape[0], right_count, right_count / X_21.shape[0]


def analysis_train_set():
    train_positive_count = 0
    train_negative_count = 0
    sum_count = 0
    for i in range(x_train.shape[0]):
        if y_train[i] == 0:
            train_negative_count = train_negative_count + 1
        else:
            train_positive_count = train_positive_count + 1
        sum_count = sum_count + 1
    print("Train Set Analysis:\nTotal number:{}\nNumber of positive samples:{}\tProportion of positive samples:{"
          "}\nNumber of negative samples:{}\tProportion of negative samples:{}\nPositive and negative sample ratio:{"
          "}\n".format(sum_count, train_positive_count, train_positive_count / sum_count, train_negative_count,
                       train_negative_count / sum_count, train_positive_count / train_negative_count))


if __name__ == "__main__":

    analysis_train_set()
    # 求均值向量
    m0 = np.array(get_mean_vector(0)).reshape(-1, 1)
    m1 = np.array(get_mean_vector(1)).reshape(-1, 1)
    s0 = get_dispersion_matrix(0, m0)
    s1 = get_dispersion_matrix(1, m1)
    sw = s0 + s1
    sb = get_sample_divergence(m0, m1)
    w_star = np.array(get_w_star(sw, m0, m1)).reshape(-1, 1)
    y0 = get_segmentation_threshold(w_star, 0)
    print("The segmentation_threshold is ", y0)
    test_sum, right_sum, accuracy = new_test_check(w_star, y0)
    print("Total specimen number:{}\nNumber of correctly predicted samples:{}\nAccuracy:{}\n".format(test_sum, right_sum, accuracy))
