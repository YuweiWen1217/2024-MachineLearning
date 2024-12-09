import numpy as np
import codecs

feature_dict = {
    "色泽": ["青绿", "乌黑", "浅白"],
    "根蒂": ["蜷缩", "稍蜷", "硬挺"],
    "敲声": ["浊响", "沉闷", "清脆"],
    "纹理": ["清晰", "稍糊", "模糊"],
    "密度": []
}
feature_types = {
    0: "discrete",
    1: "discrete",
    2: "discrete",
    3: "discrete",
    4: "discrete",
    5: "continuous",
}

lable_list = ["否", "是"]
feature_list = ["色泽", "根蒂", "敲声", "纹理", "密度"]

def load_txt(path):
    ans = []
    with codecs.open(path, "r", "GBK") as f:
        line = f.readline()
        line = f.readline()
        while line:
            d = line.rstrip("\r\n").split(',')
            re = []
            re.append(int(d[0]))
            re.append(feature_dict.get("色泽").index(d[1]) if d[1] in feature_dict.get("色泽") else float(d[1]))
            re.append(feature_dict.get("根蒂").index(d[2]) if d[2] in feature_dict.get("根蒂") else float(d[2]))
            re.append(feature_dict.get("敲声").index(d[3]) if d[3] in feature_dict.get("敲声") else float(d[3]))
            re.append(feature_dict.get("纹理").index(d[4]) if d[4] in feature_dict.get("纹理") else float(d[4]))
            re.append(float(d[5]))
            re.append(lable_list.index(d[-1]))
            ans.append(np.array(re))
            line = f.readline()
    return np.array(ans)

# 计算信息熵
def ent(Y):
    unique, counts = np.unique(Y, return_counts=True)
    prob = counts / len(Y)
    return -np.sum(prob * np.log2(prob))

# 计算信息增益
def gain_ratio(X, Y, attr):
    X_attr_col = X[:, attr]
    unique_values = np.unique(X_attr_col)
    entropy_d = ent(Y)
    weighted_entropy = 0
    split_info = 0
    for value in unique_values:
        index = np.where(X_attr_col == value)
        Y_sub = Y[index]
        weighted_entropy += len(Y_sub) / len(Y) * ent(Y_sub)
        split_info -= len(Y_sub) / len(Y) * np.log2(len(Y_sub) / len(Y))
    
    # Gain Ratio = (信息增益) / (特征划分的熵)
    gain = entropy_d - weighted_entropy
    return gain / (split_info + 1e-6)  # 防止除以0

class Node:
    def __init__(self, attr, label, v):
        self.attr = attr
        self.label = label
        self.attr_v = v
        self.children = []

# 连续属性分割点选择
def best_split_continuous(X, Y, attr):
    X_attr_col = X[:, attr]
    unique_values = np.sort(np.unique(X_attr_col))
    best_gain_ratio = -float('inf')
    best_split = None
    
    # 遍历所有可能的分割点，使用中间值作为分割点
    for i in range(len(unique_values) - 1):
        split_point = (unique_values[i] + unique_values[i + 1]) / 2
        X_left = X[X_attr_col <= split_point]
        Y_left = Y[X_attr_col <= split_point]
        X_right = X[X_attr_col > split_point]
        Y_right = Y[X_attr_col > split_point]
        
        # 计算增益比
        left_ratio = len(X_left) / len(X)
        right_ratio = len(X_right) / len(X)
        weighted_entropy = left_ratio * ent(Y_left) + right_ratio * ent(Y_right)
        split_info = -left_ratio * np.log2(left_ratio) - right_ratio * np.log2(right_ratio)
        
        # Gain Ratio
        gain_ratio_val = (ent(Y) - weighted_entropy) / (split_info + 1e-6)
        
        if gain_ratio_val > best_gain_ratio:
            best_gain_ratio = gain_ratio_val
            best_split = split_point

    return best_split, best_gain_ratio

# 决策树初始化
def dicision_tree_init(X, Y, attrs, root, purity_cal):
    """
    递归构建决策树。
    - X: 样本特征矩阵
    - Y: 样本标签
    - attrs: 剩余特征索引列表
    - root: 当前节点
    - purity_cal: 纯度计算函数（如信息增益或增益率）
    """

    # 基线条件 1: 如果当前数据集的所有样本的标签相同，直接设置为叶节点
    if len(set(Y)) == 1:
        root.attr = np.pi  # 标记为叶节点
        root.label = Y[0]  # 叶节点的分类标签
        return

    # 基线条件 2: 如果没有剩余的可用特征，设置叶节点标签为样本中出现最多的类别
    if len(attrs) == 0:
        root.attr = np.pi  # 标记为叶节点
        root.label = np.argmax(np.bincount(Y))  # 使用标签的众数作为分类结果
        return

    # 计算每个特征的纯度（如信息增益或增益率）
    purity_attrs = []  # 存储每个特征的纯度
    for i, a in enumerate(attrs):
        if feature_types[a] == "continuous":
            # 处理连续特征
            best_split, gain_ratio_val = best_split_continuous(X, Y, a)
            purity_attrs.append(gain_ratio_val)
        else:
            # 处理离散特征
            purity_attrs.append(purity_cal(X, Y, a))

    # 选择纯度最高的特征作为当前节点的划分标准
    chosen_index = purity_attrs.index(max(purity_attrs))  # 找到最大纯度的索引
    chosen_attr = attrs[chosen_index]  # 对应的特征编号

    # 设置当前节点的划分特征
    root.attr = chosen_attr
    root.label = np.pi  # 非叶节点的标签为 π，占位
    del attrs[chosen_index]  # 从剩余特征中移除已选择的特征

    # 如果是连续值特征
    if feature_types[chosen_attr] == "continuous":
        # 获取最佳分割点
        best_split, _ = best_split_continuous(X, Y, chosen_attr)
        root.attr_v = best_split  # 当前节点记录分割点

        # 将数据集划分为两个子集（<= 分割点 和 > 分割点）
        left_index = np.where(X[:, chosen_attr] <= best_split)
        right_index = np.where(X[:, chosen_attr] > best_split)
        X_left = X[left_index]
        Y_left = Y[left_index]
        X_right = X[right_index]
        Y_right = Y[right_index]

        # 创建左右子节点
        left_node = Node(-1, -1, "left")  # 左子节点标记为“left”
        right_node = Node(-1, -1, "right")  # 右子节点标记为“right”
        root.children.append(left_node)  # 添加左子节点
        root.children.append(right_node)  # 添加右子节点

        # 递归构建左右子树
        dicision_tree_init(X_left, Y_left, attrs, left_node, purity_cal)
        dicision_tree_init(X_right, Y_right, attrs, right_node, purity_cal)
    else:
        # 如果是离散值特征
        x_attr_col = X[:, chosen_attr]  # 当前特征的所有取值
        for x_v in set(x_attr_col):  # 遍历特征的所有可能取值
            n = Node(-1, -1, x_v)  # 创建子节点，标记分支特征值为 x_v
            root.children.append(n)  # 添加到当前节点的子节点列表中

            # 筛选出当前特征值等于 x_v 的子集
            index_x_equal_v = np.where(x_attr_col == x_v)
            X_x_equal_v = X[index_x_equal_v]
            Y_x_equal_v = Y[index_x_equal_v]

            # 递归构建子树
            dicision_tree_init(X_x_equal_v, Y_x_equal_v, attrs, n, purity_cal)


def dicision_tree_predict(x, tree_root):
    if tree_root.label != np.pi:
        return tree_root.label
    
    if tree_root.label == np.pi and tree_root.attr == np.pi:
        print("err!")
        return None

    chose_attr = tree_root.attr
    if feature_types[chose_attr] == "continuous":
        if x[chose_attr] <= tree_root.attr_v:
            return dicision_tree_predict(x, tree_root.children[0])
        else:
            return dicision_tree_predict(x, tree_root.children[1])
    else:
        for child in tree_root.children:
            if child.attr_v == x[chose_attr]:
                return dicision_tree_predict(x, child)
    return None

def post_pruning(tree_root, X_val, Y_val):
    if tree_root.label != np.pi:
        return tree_root  # 当前节点是叶节点，无需剪枝
    # 递归对子节点进行剪枝
    for child in tree_root.children:
        post_pruning(child, X_val, Y_val)
    # 计算剪枝前的误差
    original_error = calculate_error(tree_root, X_val, Y_val)
    # 保存当前节点信息并尝试剪枝
    original_label = tree_root.label
    tree_root.label = np.argmax(np.bincount(Y_val))  # 将当前节点转换为叶节点
    pruned_error = calculate_error(tree_root, X_val, Y_val)
    print(f"尝试剪枝节点: 属性={tree_root.attr}, 原误差={original_error:.4f}, 剪枝后误差={pruned_error:.4f}")
    # 如果剪枝后误差降低或不变，确认剪枝
    if pruned_error <= original_error:
        print(f"剪枝成功: 将属性为 {tree_root.attr} 的节点剪枝为叶节点，标签为 {tree_root.label}")
        tree_root.children = []  # 移除子树
    else:
        print(f"剪枝失败: 恢复属性为 {tree_root.attr} 的节点为非叶节点")
        tree_root.label = original_label  # 恢复为非叶节点
    return tree_root

# 计算误差
def calculate_error(tree_root, X_val, Y_val):
    predictions = []
    for i in range(X_val.shape[0]):
        x = X_val[i]
        y_pred = dicision_tree_predict(x, tree_root)
        predictions.append(y_pred)
    predictions = np.array(predictions)
    error = np.sum(predictions != Y_val) / len(Y_val)
    return error


if __name__ == '__main__':
    ans = load_txt("Watermelon-train2.csv")
    X_train = ans[:, 1: -1]
    Y_train = ans[:, -1].astype(np.int64)
    test_data = load_txt("Watermelon-test2.csv")
    X_test = test_data[:, 1:-1]
    Y_test = test_data[:, -1].astype(np.int64)
    
    r = Node(-1, -1, -1)
    attrs = [0, 1, 2, 3, 4]

    dicision_tree_init(X_train, Y_train, attrs, r, gain_ratio)

    y_predict = []
    for i in range(X_test.shape[0]):
        x = X_test[i]
        y_p = dicision_tree_predict(x, r)
        y_predict.append(y_p)

    correct_predictions = sum(1 for y_true, y_pred in zip(Y_test, y_predict) if y_true == y_pred)
    total_predictions = len(Y_test)
    accuracy = correct_predictions / total_predictions
    print('accuracy:', accuracy)


    post_pruning(r, X_test, Y_test)
    print()

    # 对测试集进行预测并计算准确率
    def calculate_accuracy(X_test, Y_test, tree):
        y_predict = []
        for i in range(X_test.shape[0]):
            x = X_test[i]
            y_p = dicision_tree_predict(x, tree)
            y_predict.append(y_p)
        correct_predictions = sum(1 for y_true, y_pred in zip(Y_test, y_predict) if y_true == y_pred)
        total_predictions = len(Y_test)
        accuracy = correct_predictions / total_predictions
        return accuracy

    accuracy2 = calculate_accuracy(X_test, Y_test, r)
    print("决策树2的准确率（剪枝后）:", accuracy2)

