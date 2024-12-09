import numpy as np
import codecs

feature_dict = {"色泽": ["青绿", "乌黑", "浅白"],
                "根蒂": ["蜷缩", "稍蜷", "硬挺"],
                "敲声": ["浊响", "沉闷", "清脆"],
                "纹理": ["清晰", "稍糊", "模糊"]
                }
lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理"]

def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\r\n").split(',')
            re = []
            # 输入编号方便追踪
            re.append(int(d[0]))
            # feature_dict.get("色泽") = ['青绿', '乌黑', '浅白']
            re.append(feature_dict.get("色泽").index(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]))
            re.append(lable_list.index(d[-1]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)

class Node:
    def __init__(self, attr, label, v):
        self.attr = attr # 非叶节点的特征名称；叶节点为pi
        self.label = label # 叶节点的分类结果；非叶节点为pi
        self.attr_v = v  # 从父节点到当前节点的分支特征值
        self.children = []  # 当前节点的子节点列表


def is_same_on_attr(X, attrs):
    '''
    数据集X在attrs上的取值是否相同
    '''
    X_a = X[:, attrs] # 从X中取出剩余特征的部分
    target = X_a[0]
    for r in range(X_a.shape[0]):
        row = X_a[r]
        # 第一个样本和任意一个样本中，任意一个特征不同就返回false
        if (row != target).any():
            return False
    # 至此，全部比较完毕，发现所有样本特征都相同，返回true
    return True

def ent(D):
    # D: 样本标签列表
    # ENT(D) = - Σ P(k) * log2(P(k))
    s = 0
    for k in set(D):
        p_k = np.sum(np.where(D == k, 1, 0)) / np.shape(D)[0]
        if p_k == 0:
            # 此时Pklog2Pk 定义为 0
            continue
        s += p_k * np.log2(p_k)
    return -s

def gain(X, Y, attr):
    # return: 总的熵 - 各个分支的熵的加权平均值
    # X, Y是样本及标签， attr是某个特征
    x_attr_col = X[:, attr]
    ent_Dv = []
    weight_Dv = []
    # 离散值处理
    for x_v in set(x_attr_col):
        index_x_equal_v = np.where(x_attr_col == x_v) # 特征attr的值等于x_v的样本编号
        y_x_equal_v = Y[index_x_equal_v]    # 特征attr的值等于x_v的样本类别
        ent_Dv.append(ent(y_x_equal_v))
        weight_Dv.append(np.shape(y_x_equal_v)[0] / np.shape(Y)[0])
    return ent(Y) - np.sum(np.array(ent_Dv) * np.array(weight_Dv))

def dicision_tree_init(X, Y, attrs, root, purity_cal):
    """
    递归构建决策树的初始化函数。
    - X、Y 训练集
    - attrs: 剩余特征列表（索引）
    - root: 当前所在的节点
    - purity_cal: 计算信息增益
    """

    # 递归基 1: 如果当前数据集 Y 中所有样本的标签相同，则直接设为叶节点
    if len(set(Y)) == 1:
        root.attr = np.pi
        root.label = Y[0]
        return None

    # 递归基 2: 如果特征用尽，或者数据集在剩余特征上取值相同
    if len(attrs) == 0 or is_same_on_attr(X, attrs):
        root.attr = np.pi
        # Y中出现最多的标签值为叶节点标签
        root.label = np.argmax(np.bincount(Y))
        return None

    # 计算每个特征的划分收益（如信息增益）
    purity_attrs = []
    for i, a in enumerate(attrs):
        # 计算当前特征 a 的纯度值
        p = purity_cal(X, Y, a)
        purity_attrs.append(p)

    # 选择纯度最大的特征作为划分特征
    chosen_index = purity_attrs.index(max(purity_attrs))  # 最大纯度的特征索引
    chosen_attr = attrs[chosen_index]  # 对应的特征编号

    # 设置当前节点为非叶节点，设置其特征
    root.attr = chosen_attr
    root.label = np.pi

    # 从剩余特征中移除已选择的特征
    del attrs[chosen_index]

    # 获取当前划分特征列
    x_attr_col = X[:, chosen_attr]

    # 对当前特征的每个取值，生成对应的子节点
    for x_v in set(x_attr_col):  # 遍历划分特征的所有取值
        # 创建新的子节点，初始化为 -1 表示暂未设置具体值
        n = Node(-1, -1, x_v)  # attr = -1, label = -1, attr_v = x_v
        root.children.append(n)  # 将子节点添加到当前节点的子节点列表
        # 不可能Dv empty 要是empty压根不会在set里
        # 选出 X[attr] == x_v的行

        # 筛选出在该特征取值下的样本及对应标签
        index_x_equal_v = np.where(x_attr_col == x_v)  # 筛选条件
        X_x_equal_v = X[index_x_equal_v]  # 筛选后的特征子集
        Y_x_equal_v = Y[index_x_equal_v]  # 筛选后的标签子集

        # 递归调用构建子树
        dicision_tree_init(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)


def dicision_tree_predict(x, tree_root):
    # 到叶节点了，返回标签
    if tree_root.label != np.pi:
        return tree_root.label

    # 一种错误情况
    if tree_root.label == np.pi and tree_root.attr == np.pi:
        print("err!")
        return None

    # 选择当前节点的划分特征
    chose_attr = tree_root.attr
    for child in tree_root.children:
        if child.attr_v == x[chose_attr]:
            return dicision_tree_predict(x, child)
    return None

if __name__ == '__main__':
    ans = load_txt("Watermelon-train1.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1].astype(np.int64)
    test_data = load_txt("Watermelon-test1.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1].astype(np.int64)
    r = Node(-1, -1, -1)
    attrs = [0, 1, 2, 3]

    dicision_tree_init(X_train, Y_train, attrs, r, gain)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, r)
        y_predict.append(y_p)

    # 计算准确率
    correct_predictions = sum(1 for y_true, y_pred in zip(Y_test, y_predict) if y_true == y_pred)
    total_predictions = len(Y_test)
    accuracy = correct_predictions / total_predictions
    print('accuracy:', accuracy)