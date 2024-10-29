def watermelon2():
    train_data = [
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '是'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '是'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '否'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '否'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '否']
    ]

    test_data = [
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '是'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '否'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '否'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '否'],
    ]

    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']

    return train_data, test_data, labels


class TreeNode:
    """
    决策树结点类
    """
    current_index = 0

    def __init__(self, parent=None, attr_name=None, children=None, judge=None, split=None, data_index=None,
                 attr_value=None, rest_attribute=None):
        """
        决策树结点类初始化方法
        :param parent: 父节点
        """
        self.parent = parent  # 父节点，根节点的父节点为 None
        self.attribute_name = attr_name  # 本节点上进行划分的属性名
        self.attribute_value = attr_value  # 本节点上划分属性的值，是与父节点的划分属性名相对应的
        self.children = children  # 孩子结点列表
        self.judge = judge  # 如果是叶子结点，需要给出判断
        self.split = split  # 如果是使用连续属性进行划分，需要给出分割点
        self.data_index = data_index  # 对应训练数据集的训练索引号
        self.index = TreeNode.current_index  # 当前结点的索引号，方便输出时查看
        self.rest_attribute = rest_attribute  # 尚未使用的属性列表
        TreeNode.current_index += 1

    def to_string(self):
        """用一个字符串来描述当前结点信息"""
        this_string = 'current index : ' + str(self.index) + ";\n"

        if not (self.parent is None):
            parent_node = self.parent
            this_string = this_string + 'parent index : ' + str(parent_node.index) + ";\n"
            this_string = this_string + str(parent_node.attribute_name) + " : " + str(self.attribute_value) + ";\n"

        this_string = this_string + "data : " + str(self.data_index) + ";\n"

        if not (self.children is None):
            this_string = this_string + 'select attribute is : ' + str(self.attribute_name) + ";\n"
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            this_string = this_string + 'children : ' + str(child_list)

        if not (self.judge is None):
            this_string = this_string + 'label : ' + self.judge

        return this_string

    def print_dot(self):
        dot_string = ''
        if not (self.attribute_name is None):
            if not(self.children is None):
                dot_string = dot_string + str(self.attribute_name)+'[label='+'"'+str(self.attribute_name)+'"'+'];'+'\n'

        if not (self.judge is None):
            parent_node=self.parent
            dot_string = dot_string + str(self.index)+'[label='+'"'+str(self.judge)+'"'+'];'+'\n'
            name = str(self.judge)
            dot_string = dot_string + str(parent_node.attribute_name) + '->' + str(self.index)+ \
                         '[label=' + '"' + str(self.attribute_value) + '"' + '];' + '\n'

        if not (self.parent is None):
            if not (self.children is None):
                parent_node=self.parent
                name = ''
                if not(self.attribute_name is None):
                    name=str(self.attribute_name)
                    dot_string = dot_string + str(parent_node.attribute_name)+'->'+name+\
                            '[label='+'"'+str(self.attribute_value)+'"'+'];'+'\n'

        return dot_string