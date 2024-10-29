# CART决策树，使用基尼指数（Gini index）来选择划分属性
# 分别实现预剪枝、后剪枝和不进行剪枝的实现

import math
from lib.lib import TreeNode
from lib.lib import watermelon2
import graphviz

def is_number(s):
    """判断一个字符串是否为数字，如果是数字，我们认为这个属性的值是连续的"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def ent(labels=[]):
    """
    计算数据集的信息熵
    :param labels: 数据集的类别标签
    :return:
    """
    label_name=[]
    label_count=[]
    entropy = 0


    for item in labels:
        if not (item in label_name):
            label_name.append(item)
            label_count.append(1)
        else:
            index=label_name.index(item)
            label_count[index] +=1

    n=sum(label_count)
    entropy=0.0
    for item in label_count:
        p=item/n
        entropy = entropy-p*math.log(p,2)

    return entropy


def gain(attribute, labels, is_value=False):
    """
    计算信息增益
    :param attribute: 集合中样本该属性的值列表
    :param labels: 集合中样本的数据标签
    :return:
    """
    # is_value = False  # 当前属性是离散的形容词还是连续的数值
    info_gain = ent(labels)
    n = len(labels)
    split_value = None  # 如果是连续值的话，也需要返回分隔界限的值

    if is_value:
        # print('attribute', attribute)
        # 属性值是连续的数值，首先应该使用二分法寻找最佳分割点
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        split = []  # 候选的分隔点
        for i in range(0, n - 1):
            temp = (sorted_attribute[i] + sorted_attribute[i + 1]) / 2
            split.append(temp)
        info_gain_list = []
        # print('split', split)
        for temp_split in split:
            low_labels = []
            high_labels = []
            for i in range(0, n):
                if attribute[i] <= temp_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            temp_gain = info_gain - len(low_labels) / n * ent(low_labels) - len(high_labels) / n * ent(high_labels)
            info_gain_list.append(temp_gain)

        # print('info_gain_list', info_gain_list)
        info_gain = max(info_gain_list)
        max_index = info_gain_list.index(info_gain)
        split_value = split[max_index]
    else:
        # 属性值是离散的值
        attribute_dict = {}
        label_dict = {}
        index = 0
        for item in attribute:
            if attribute_dict.__contains__(item):
                attribute_dict[item] = attribute_dict[item] + 1
                label_dict[item].append(labels[index])
            else:
                attribute_dict[item] = 1
                label_dict[item] = [labels[index]]
            index += 1

        for key, value in attribute_dict.items():
            info_gain = info_gain - value / n * ent(label_dict[key])

    return info_gain, split_value


def finish_node(current_node, data, label):
    """
    完成当前结点的后续计算，包括选择属性，划分子节点等
    :param current_node: 当前的结点
    :param data: 数据集
    :param label: 数据集的 label
    :param rest_title: 剩余的可用属性名
    :return:
    """
    n = len(label)

    # 判断当前结点的数据是否属于同一类，如果是，直接标记为叶子结点并返回
    one_class = True

    this_data_index = current_node.data_index
    for i in this_data_index:
        for j in this_data_index:
            if label[i] != label[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        current_node.judge = label[this_data_index[0]]
        print(str(current_node.parent.attribute_name)+': '+str(current_node.attribute_value))
        print('包含的数据索引:'+str(current_node.data_index)+'均属于同一标签，是叶子结点')
        return

    rest_title = current_node.rest_attribute  # 候选属性
    if len(rest_title) == 0:  # 如果候选属性为空，则是个叶子结点。需要选择最多的那个类作为该结点的类
        label_count = {}
        temp_data = current_node.data_index
        for index in temp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count)
        current_node.judge = final_label

        print('包含的数据索引:'+str(current_node.data_index)+'候选属性为空，是叶子结点')
        return

    title_gain = {}  # 记录每个属性的信息增益
    title_split_value = {}  # 记录每个属性的分隔值，如果是连续属性则为分隔值，如果是离散属性则为None
    for title in rest_title:
        attr_values = []
        current_label = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            current_label.append(label[index])
        temp_data = data[0]
        this_gain, this_split_value = gain(attr_values, current_label, is_number(temp_data[title]))  # 如果属性值为数字，则认为是连续的
        title_gain[title] = this_gain
        title_split_value[title] = this_split_value

    best_attr = max(title_gain, key=title_gain.get)  # 信息增益最大的属性名
    current_node.attribute_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if is_number(a_data[best_attr]):  # 如果是该属性的值为连续数值
        split_value = title_split_value[best_attr]
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr] <= split_value:
                small_data.append(index)
            else:
                large_data.append(index)
        small_str = '<=' + str(split_value)
        large_str = '>' + str(split_value)
        small_child = TreeNode(parent=current_node, data_index=small_data, attr_value=small_str,
                               rest_attribute=rest_title.copy())
        large_child = TreeNode(parent=current_node, data_index=large_data, attr_value=large_str,
                               rest_attribute=rest_title.copy())
        current_node.children = [small_child, large_child]

    else:  # 如果该属性的值是离散值
        best_titlevalue_dict = {}  # key是属性值的取值，value是个list记录所包含的样本序号
        for index in current_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                temp_list = best_titlevalue_dict[this_data[best_attr]]
                temp_list.append(index)
            else:
                temp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = temp_list

        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode(parent=current_node, data_index=index_list, attr_value=key,
                               rest_attribute=rest_title.copy())
            children_list.append(a_child)
        current_node.children = children_list

    # print(current_node.to_string())
    for child in current_node.children:  # 递归
        finish_node(child, data, label)


def id3_tree(Data, title, label):
    """
    id3方法构造决策树，使用的标准是信息增益（信息熵）
    :param Data: 数据集，每个样本是一个 dict(属性名：属性值)，整个 Data 是个大的 list
    :param title: 每个属性的名字，如 色泽、含糖率等
    :param label: 存储的是每个样本的类别
    :return:
    """
    n = len(Data)
    rest_title = title.copy()
    root_data = []
    for i in range(0, n):
        root_data.append(i)

    root_node = TreeNode(data_index=root_data, rest_attribute=title.copy())
    finish_node(root_node, Data, label)

    return root_node


def print_tree(root=TreeNode()):
    """
    打印输出一颗树
    :param root: 根节点
    :return:
    """
    node_list = [root]
    while (len(node_list) > 0):
        current_node = node_list[0]
        print('--------------------------------------------')
        print(current_node.to_string())
        print('--------------------------------------------')
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)


def print_tree(root):
    """
    打印输出一颗树
    :param root: 根节点
    :return:
    """
    node_list = [root]
    while (len(node_list) > 0):
        current_node = node_list[0]
        print('--------------------------------------------')
        print(current_node.to_string())
        print(current_node.print_dot())
        print('--------------------------------------------')
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)

def dot_graph(root):
    string = []
    node_list = [root]
    Note = open('../test.dot', mode='w')
    Note.write('digraph G {	node [fontname=FangSong];edge[fontname=FangSong];')

    while (len(node_list) > 0):
        current_node = node_list[0]
#        string.append(current_node.print_dot())
        Note = open('../test.dot', mode='a')
        Note.write(current_node.print_dot())
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)

    Note = open('../test.dot', mode='a')
    Note.write('}')

def dot_graph_draw():
    with open('../test.dot', encoding='gbk') as fj:
        source = fj.read()

    dot = graphviz.Source(source)
    dot.view()

def classify_data(decision_tree, x={}):
    """
    使用决策树判断一个数据样本的类别标签
    :param decision_tree: 决策树的根节点
    :param x: 要进行判断的样本
    :return:
    """
    current_node = decision_tree
    while current_node.judge is None:
        if current_node.split is None:  # 离散属性
            can_judge = False  # 如果训练数据集不够大，测试数据集中可能会有在训练数据集中没有出现过的属性值
            for child in current_node.children:
                if child.attribute_value == x[current_node.attribute_name]:
                    current_node = child
                    can_judge = True
                    break
            if not can_judge:
                return None
        else:
            child_list = current_node.children
            if x[current_node.attribute_name] <= current_node.split:
                current_node = child_list[0]
            else:
                current_node = child_list[1]

    return current_node.judge


def run_test():
    train_watermelon, test_watermelon, title = watermelon2()

    # 先处理数据
    train_data = []
    test_data = []
    train_label = []
    test_label = []

    for melon in train_watermelon:
        a_dict = {}
        dim = len(melon) - 1
        for i in range(0, dim):
            a_dict[title[i]] = melon[i]
        train_data.append(a_dict)
        train_label.append(melon[dim])

    for melon in test_watermelon:
        a_dict = {}
        dim = len(melon) - 1
        for i in range(0, dim):
            a_dict[title[i]] = melon[i]
        test_data.append(a_dict)
        test_label.append(melon[dim])

    decision_tree = id3_tree(train_data, title, train_label)
    print_tree(decision_tree)
    dot_graph(decision_tree)
    dot_graph_draw()

    test_judge = []
    for melon in test_data:
        test_judge.append(classify_data(decision_tree, melon))
    print('决策树在测试数据集上的分类结果是：', test_judge)
    print('测试数据集的正确类别信息应该是：  ', test_label)

    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == test_judge[i]:
            accuracy += 1
    accuracy /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：' + str(accuracy * 100) + "%")

if __name__ == '__main__':
    run_test()