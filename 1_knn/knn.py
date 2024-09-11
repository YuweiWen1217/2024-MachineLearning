import numpy as np
from collections import Counter
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score

# 距离计算：欧氏距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# kNN算法实现，处理多个k值
def k_nearest_neighbors(train_data, train_labels, test_sample, k_values):
    distances = []
    # 计算测试样本与每个训练样本之间的距离
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_sample)
        distances.append((dist, train_labels[i]))
    
    # 根据距离进行排序
    distances.sort(key=lambda x: x[0])
    
    predictions_for_k = []
    # 遍历 k_values 中的 k 值，获取每个 k 值对应的分类结果
    for k in k_values:
        # 选择 k 个最近的样本
        k_neighbors = [distances[i][1] for i in range(k)]
        # 投票选择出现最多的类别
        most_common = Counter(k_neighbors).most_common(1)
        predictions_for_k.append(most_common[0][0])
    
    return predictions_for_k

# 进行kNN分类，并使用留一法进行验证
def knn_with_loocv(data, labels, k_values):
    loo = LeaveOneOut()
    # 初始化一个字典，存储每个k值对应的预测结果
    predictions_dict = {k: [] for k in k_values}
    
    # 留一法遍历数据
    for train_index, test_index in loo.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        
        # 调用k_nearest_neighbors函数，并获取每个k值对应的预测
        predictions_for_k = k_nearest_neighbors(train_data, train_labels, test_data[0], k_values)
        
        # 将每个k值对应的预测存储到对应的列表中
        for i, k in enumerate(k_values):
            predictions_dict[k].append(predictions_for_k[i])
    
    # 计算每个k值的准确率
    accuracies = []
    for k in k_values:
        accuracy = accuracy_score(labels, predictions_dict[k])
        accuracies.append(accuracy)
    
    return accuracies

# 加载semeion.data数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    # 前256列是特征，后10列是one-hot编码的标签
    X = data[:, :256]
    # 将one-hot编码转换为单一的数字标签
    y = np.argmax(data[:, 256:], axis=1)
    return X, y

# 加载数据
file_path = 'semeion.data'
X, y = load_semeion_data(file_path)
# 设定不同的k值
k_values = [5, 9, 13]
# 传入k值数组，得到每个k值的精度
accuracies = knn_with_loocv(X, y, k_values)
# 输出每个k值对应的准确率
for k, acc in zip(k_values, accuracies):
    print(f'k={k} 时的精度: {acc:.4f}')
