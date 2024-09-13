import numpy as np
from collections import Counter
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score
from sklearn.metrics import normalized_mutual_info_score
from pycm import ConfusionMatrix
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import entropy


# 计算欧氏距离、NMI (归一化互信息)、CEN (混淆熵)
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def calculate_nmi(labels, predictions):
    return normalized_mutual_info_score(labels, predictions)
# 计算
def calculate_cen(labels, predictions):
    cm = ConfusionMatrix(actual_vector=labels, predict_vector=predictions)
    cen = sum(cm.CEN.values()) / len(cm.CEN)
    return cen

def calculate_cen1(y_true, y_pred):
    matrix = np.zeros((10, 10))
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[true_label][pred_label] += 1
    matrix /= len(y_true)
    rentropy = [entropy(row) for row in matrix if np.sum(row) > 0]
    cen = np.mean(rentropy)
    return cen

# 自己实现多k值的kNN算法
def k_nearest_neighbors(train_data, train_labels, test_sample, k_values):
    distances = []
    # 计算测试样本与每个训练样本之间的距离并排序
    for i in range(len(train_data)):
        dist = euclidean_distance(train_data[i], test_sample)
        distances.append((dist, train_labels[i]))
    distances.sort(key=lambda x: x[0])
    predictions = [] # 储存该测试样本的三个预测值
    # 遍历 k_values 中的 k 值，获取每个 k 值对应的分类结果
    for k in k_values:
        k_neighbors = [distances[i][1] for i in range(k)]
        most_common = Counter(k_neighbors).most_common(1)
        predictions.append(most_common[0][0])
    return predictions

# 进行kNN分类，并使用留一法进行验证
def knn_with_loocv(data, labels, k_values):
    loo = LeaveOneOut()
    # 初始化一个字典，存储每个k值对应的预测结果
    predictions_dict = {k: {'correct':[], 'predictions':[], 'true_labels':[]} for k in k_values}
    
    # 留一法遍历数据
    for train_index, test_index in loo.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]
        predictions = k_nearest_neighbors(train_data, train_labels, test_data[0], k_values)
        
        # 将每个k值对应的预测存储到对应的列表中
        for i, k in enumerate(k_values):
            predictions_dict[k]['correct'].append(int(predictions[i] == test_labels[0]))
            predictions_dict[k]['predictions'].append(predictions[i])
            predictions_dict[k]['true_labels'].append(test_labels[0])
    
    # 计算每个k值的平均值
    accuracies = []
    for k in k_values:
        accuracy = np.mean(predictions_dict[k]['correct'])
        nmi = calculate_nmi(predictions_dict[k]['true_labels'], predictions_dict[k]['predictions'])
        cen = calculate_cen(predictions_dict[k]['true_labels'], predictions_dict[k]['predictions'])
        accuracies.append((accuracy, nmi, cen))
    return accuracies

# 加载semeion.data数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    # 前256列是特征，后10列是one-hot编码的标签
    X = data[:, :256]
    # 将one-hot编码转换为单一的数字标签
    y = np.argmax(data[:, 256:], axis=1)
    return X, y

#################################
# sklearn
#################################
# 使用Scikit-learn的kNN分类器进行对比
def compare_with_sklearn_knn(X, y, k_values):
    accuracies = []
    nmis = []
    cens = []

    # 遍历不同的k值
    for k in k_values:
        # 使用Sklearn的KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=k)
        loo = LeaveOneOut()
        
        predictions = []
        true_labels = []
        
        for train_index, test_index in loo.split(X):
            train_data, test_data = X[train_index], X[test_index]
            train_labels, test_labels = y[train_index], y[test_index]
            
            # 拟合并预测
            knn.fit(train_data, train_labels)
            prediction = knn.predict(test_data)
            predictions.append(prediction[0])
            true_labels.append(test_labels[0])
        
        # 计算 ACC, NMI, CEN
        acc = accuracy_score(true_labels, predictions)
        nmi = calculate_nmi(true_labels, predictions)
        cen = calculate_cen(true_labels, predictions)
        
        accuracies.append(acc)
        nmis.append(nmi)
        cens.append(cen)
    
    return accuracies, nmis, cens


print("自行实现kNN")
# 加载数据
file_path = 'semeion.data'
X, y = load_semeion_data(file_path)
# 设定不同的k值
k_values = [5, 9, 13]
# 传入k值数组，得到每个k值的精度
results = knn_with_loocv(X, y, k_values)
# 输出每个k值对应的准确率
for i, k in enumerate(k_values):
    accuracy, nmi, cen = results[i]
    print(f'k={k} 时的精度: {accuracy:.4f}, NMI: {nmi:.4f}, CEN: {cen:.4f}')

print("sklearn实现kNN")
# 调用对比函数并输出结果
sklearn_accuracies, sklearn_nmis, sklearn_cens = compare_with_sklearn_knn(X, y, k_values)
# 输出Sklearn kNN分类器的结果
for i, k in enumerate(k_values):
    print(f'k={k} 时的精度: {sklearn_accuracies[i]:.4f}, NMI: {sklearn_nmis[i]:.4f}, CEN: {sklearn_cens[i]:.4f}')

