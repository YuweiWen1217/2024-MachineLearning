import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors

# 设置参数：样本数量、均值、协方差矩阵
N = 1000
mu_1 = np.array([1, 4])
mu_2 = np.array([4, 1])
mu_3 = np.array([8, 4])
cov = 2 * np.eye(2)


# 生成数据集 X1 函数
def generate_X1(N):
    # labels:数据点对应的分布；X1：数据点
    labels = np.random.choice([0, 1, 2], size=N)  
    X1 = np.zeros((N, 2))
    for i in range(N):
        # 生成不同均值、方差为2的二维随机向量
        if labels[i] == 0:
            X1[i] = np.random.multivariate_normal(mu_1, cov)
        elif labels[i] == 1:
            X1[i] = np.random.multivariate_normal(mu_2, cov)
        else:
            X1[i] = np.random.multivariate_normal(mu_3, cov)
    return X1, labels

# 生成数据集 X2 函数
def generate_X2(N):
    # 定义先验概率
    prior_probs = [0.6, 0.3, 0.1]
    labels = np.random.choice([0, 1, 2], size=N, p=prior_probs)  
    X2 = np.zeros((N, 2))
    for i in range(N):
        # 生成不同均值、方差为2的二维随机向量
        if labels[i] == 0:
            X2[i] = np.random.multivariate_normal(mu_1, cov)
        elif labels[i] == 1:
            X2[i] = np.random.multivariate_normal(mu_2, cov)
        else:
            X2[i] = np.random.multivariate_normal(mu_3, cov)
    return X2, labels

# 生成两个数据集
X1, labels1 = generate_X1(N)
X2, labels2 = generate_X2(N)

# 可视化两个数据集
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X1[:, 0], X1[:, 1], c=labels1, cmap='viridis', alpha=0.5)
plt.title('DataSet-X1')
plt.xlabel('x1')
plt.ylabel('x2')

plt.subplot(1, 2, 2)
plt.scatter(X2[:, 0], X2[:, 1], c=labels2, cmap='viridis', alpha=0.5)
plt.title('DataSet-X2')
plt.xlabel('x1')
plt.ylabel('x2')

plt.tight_layout()
plt.show()



# 计算x在三种正态分布下的概率密度
def density(x):
    return [
        multivariate_normal.pdf(x, mean=mu_1, cov=cov),
        multivariate_normal.pdf(x, mean=mu_2, cov=cov),
        multivariate_normal.pdf(x, mean=mu_3, cov=cov)
    ]

# 进行分类
def classify(X, dataset = '1'):
    predictions_MLE = []
    predictions_MAP = []
    for x in X:
        # densities：一个向量，x在三种正态分布下的概率密度。
        densities = density(x)
        # 似然率测试规则：直接选择P(X|\theta)最大者
        predictions_MLE.append(np.argmax(densities))
        # 最大后验概率规则：选择P(\theta|X)最大者
        if dataset == '1':
            priors = [1/3, 1/3, 1/3]
        else:
            priors = [0.6, 0.3, 0.1]
        posterior = [densities[i] * priors[i] for i in range(3)]
        predictions_MAP.append(np.argmax(posterior))
    return predictions_MLE, predictions_MAP

# 计算错误率
def calculate_error_rate(predictions, true_labels):
    return np.mean(np.array(predictions) != np.array(true_labels))

print("初级要求")

# 应用两种方法进行分类
predictions_X1_MLE, predictions_X1_MAP = classify(X1, dataset ='1') 
predictions_X2_MLE, predictions_X2_MAP = classify(X2, dataset ='2')

# 计算错误率
error_rate_X1_MLE = calculate_error_rate(predictions_X1_MLE, labels1)
error_rate_X1_MAP = calculate_error_rate(predictions_X1_MAP, labels1)

error_rate_X2_MLE = calculate_error_rate(predictions_X2_MLE, labels2)
error_rate_X2_MAP = calculate_error_rate(predictions_X2_MAP, labels2)

# 输出结果
print("X1数据集的分类错误率:")
print(f"似然率测试规则: {error_rate_X1_MLE:.4f}")
print(f"最大后验概率规则: {error_rate_X1_MAP:.4f}")

print("\nX2数据集的分类错误率::")
print(f"似然率测试规则: {error_rate_X2_MLE:.4f}")
print(f"最大后验概率规则: {error_rate_X2_MAP:.4f}")


# 高斯核函数，计算样本x和数据点xi之间的高斯核值，h为窗口宽度。
def gaussian_kernel(x, x_i, h):
    return np.exp(-0.5 * np.sum((x - x_i) ** 2) / h**2) / (h * np.sqrt(2 * np.pi))

# x的核函数密度概率估计
def kernel_density_estimate(X, x, h):
    return np.mean([gaussian_kernel(x, x_i, h) for x_i in X])


# 分类
def classify_with_kernel(X, labels, h, dataset):
    predictions_MLE = []
    predictions_MAP = []
    for idx, x in enumerate(X):
        # 生成训练集，排除当前样本
        X_train = np.delete(X, idx, axis=0)
        labels_train = np.delete(labels, idx)
        # 计算每个类别的核密度估计
        densities = []
        for i in range(3):
            class_samples = X_train[labels_train == i]
            density = kernel_density_estimate(class_samples, x, h)
            densities.append(density)

        # 重复初级要求的分类过程
        predictions_MLE.append(np.argmax(densities))
        if dataset == '1':
            priors = [1/3, 1/3, 1/3]
        else:
            priors = [0.6, 0.3, 0.1]
        posterior = [densities[i] * priors[i] for i in range(3)]
        predictions_MAP.append(np.argmax(posterior))
    return predictions_MLE, predictions_MAP



h_values = [0.1, 0.5, 1, 1.5, 2]
results_X1_MLE = {}
results_X2_MLE = {}
results_X1_MAP = {}
results_X2_MAP = {}

for h in h_values:
    # X1 分类
    predictions_X1_MLE, predictions_X1_MAP = classify_with_kernel(X1, labels1, h, '1')
    error_rate_X1_MLE = calculate_error_rate(predictions_X1_MLE, labels1)
    error_rate_X1_MAP = calculate_error_rate(predictions_X1_MAP, labels1)
    results_X1_MLE[h] = error_rate_X1_MLE
    results_X1_MAP[h] = error_rate_X1_MAP

    # X2 分类
    predictions_X2_MLE, predictions_X2_MAP = classify_with_kernel(X2, labels1, h, '2')
    error_rate_X2_MLE = calculate_error_rate(predictions_X2_MLE, labels2)
    error_rate_X2_MAP = calculate_error_rate(predictions_X2_MAP, labels2)
    results_X2_MLE[h] = error_rate_X2_MLE
    results_X2_MAP[h] = error_rate_X2_MAP

# 输出错误率结果
print("数据集合 X1 的错误率:")
for h in h_values:
    print(f"h={h}: 最大似然估计错误率 = {results_X1_MLE[h]:.4f}, 最大后验概率估计错误率 = {results_X1_MAP[h]:.4f}")

print("\n数据集合 X2 的错误率:")
for h in h_values:
    print(f"h={h}: 最大似然估计错误率 = {results_X2_MLE[h]:.4f}, 最大后验概率估计错误率 = {results_X2_MAP[h]:.4f}")

# 可视化输出
plt.figure(figsize=(12, 6))

# X1 错误率可视化
plt.subplot(1, 2, 1)
plt.plot(h_values, list(results_X1_MLE.values()), marker='o', label='MLE')
plt.plot(h_values, list(results_X1_MAP.values()), marker='x', label='MAP')
plt.title('DataSet-X1')
plt.xlabel('h')
plt.ylabel('error rate')
plt.xticks(h_values)
plt.legend()

# X2 错误率可视化
plt.subplot(1, 2, 2)
plt.plot(h_values, list(results_X2_MLE.values()), marker='o', label='MLE')
plt.plot(h_values, list(results_X2_MAP.values()), marker='x', label='MAP')
plt.title('DataSet-X2')
plt.xlabel('h')
plt.ylabel('error rate')
plt.xticks(h_values)
plt.legend()

plt.tight_layout()
plt.show()



# knn概率密度估计函数
def knn_density_estimate(X, k, grid_x, grid_y):
    n = X.shape[0]
    # 创建网格点坐标
    grid_points = np.array([[gx, gy] for gx in grid_x for gy in grid_y])
    # 找到最近的第k个点的距离和索引，用这个点计算面积
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(grid_points)
    volumes = (np.pi * distances[:, k - 1] ** 2)
    # P = k/nS
    density_estimates = np.zeros(len(grid_points))
    density_estimates[distances[:, k - 1] == 0] = 1
    non_zero_volumes = volumes > 0
    density_estimates[non_zero_volumes] = k / (n * volumes[non_zero_volumes]) 
    density_estimates[density_estimates > 1] = 1
    return grid_points, density_estimates

# 预设 k 值
k_values = [1, 3, 5]
num_distributions = 3

# 概率密度可视化
fig = plt.figure(figsize=(18, 12))

# 创建网格
x_min_X1, x_max_X1 = X1[:, 0].min() - 1, X1[:, 0].max() + 1
y_min_X1, y_max_X1 = X1[:, 1].min() - 1, X1[:, 1].max() + 1
x_min_X2, x_max_X2 = X2[:, 0].min() - 1, X2[:, 0].max() + 1
y_min_X2, y_max_X2 = X2[:, 1].min() - 1, X2[:, 1].max() + 1

grid_x_X1 = np.linspace(x_min_X1, x_max_X1, 100)
grid_y_X1 = np.linspace(y_min_X1, y_max_X1, 100)

grid_x_X2 = np.linspace(x_min_X2, x_max_X2, 100)
grid_y_X2 = np.linspace(y_min_X2, y_max_X2, 100)

for j, k in enumerate(k_values):

    # 绘制 X1 数据集的概率密度分布图
    ax1 = fig.add_subplot(2, 3, j + 1, projection='3d')
    grid_points, density_X1 = knn_density_estimate(X1, k, grid_x_X1, grid_y_X1)
    grid_xx, grid_yy = np.meshgrid(grid_x_X1, grid_y_X1)
    density_surface_X1 = density_X1.reshape((len(grid_x_X1), len(grid_y_X1)))
    ax1.plot_surface(grid_xx, grid_yy, density_surface_X1, cmap='viridis', alpha=0.7)
    ax1.set_title(f'X1 - k={k}')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('Probability density estimation')

    # 绘制 X2 数据集的概率密度分布图
    ax2 = fig.add_subplot(2, 3, j + 4, projection='3d')
    grid_points, density_X2 = knn_density_estimate(X2, k, grid_x_X2, grid_x_X2)
    grid_xx, grid_yy = np.meshgrid(grid_x_X2, grid_y_X2)
    density_surface_X2 = density_X2.reshape((len(grid_x_X2), len(grid_x_X2)))
    ax2.plot_surface(grid_xx, grid_yy, density_surface_X2, cmap='viridis', alpha=0.7)
    ax2.set_title(f'X2 - k={k}')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('Probability density estimation')

plt.tight_layout()
plt.show()