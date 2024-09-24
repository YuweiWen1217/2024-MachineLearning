import numpy as np
from collections import Counter
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.ndimage import rotate
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical


#################################
# 计算欧氏距离、NMI (归一化互信息)、CEN (混淆熵)
#################################
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
def calculate_nmi(labels, predictions):
    return normalized_mutual_info_score(labels, predictions)
def calculate_cen(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    probs = cm / np.sum(cm)
    cen = -np.nansum(probs * np.log(probs + 1e-10))
    return cen


#################################
# 数据加载
#################################
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    # 前256列是特征，后10列是one-hot编码的标签
    X = data[:, :256]
    # 将one-hot编码转换为单一的数字标签
    y = np.argmax(data[:, 256:], axis=1)
    return X, y


#################################
# 自己实现多k值的kNN算法
#################################
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



# 数据增强：对图像进行左上、左下旋转
def augment_data(X, y):
    X_rotated_left_up = np.array([rotate(x.reshape(16, 16), angle=20, reshape=False).flatten() for x in X])
    X_rotated_left_down = np.array([rotate(x.reshape(16, 16), angle=-20, reshape=False).flatten() for x in X])
    
    # 将原始数据和增强数据拼接
    X_augmented = np.concatenate([X, X_rotated_left_up, X_rotated_left_down], axis=0)
    y_augmented = np.concatenate([y, y, y], axis=0)
    
    return X_augmented, y_augmented

# 构建CNN模型
def build_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


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

print("数据加强后使用CNN进行分类")
X_augmented, y_augmented = augment_data(X, y)
# 预处理数据以适应CNN
X_augmented = X_augmented.reshape(-1, 16, 16, 1)  # 将数据重塑为适合CNN输入的格式
y_augmented_categorical = to_categorical(y_augmented)  # 将标签转换为one-hot编码
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_augmented, y_augmented_categorical, test_size=0.2, random_state=42)
# 构建CNN模型
input_shape = (16, 16, 1)
num_classes = 10  # 手写数字类别数
model = build_cnn(input_shape, num_classes)
# 训练模型
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 测试模型
y_test_labels = np.argmax(y_test, axis=1)  # 转换one-hot编码为标签
y_pred = np.argmax(model.predict(X_test), axis=1)  # 模型预测标签
# 计算准确率
accuracy = accuracy_score(y_test_labels, y_pred)
# 计算NMI和CEN
nmi = calculate_nmi(y_test_labels, y_pred)
cen = calculate_cen(y_test_labels, y_pred)
# 输出结果
print(f'准确率: {accuracy:.4f}')
print(f'NMI: {nmi:.4f}')
print(f'CEN: {cen}')