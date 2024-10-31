import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neighbors import NearestNeighbors

# 设置参数
N = 1000  # 每个数据集的样本数量
mu_1 = np.array([1, 4])  # 第一个分布的均值向量
mu_2 = np.array([4, 1])  # 第二个分布的均值向量
mu_3 = np.array([8, 4])  # 第三个分布的均值向量
cov = 2 * np.eye(2)  # 协方差矩阵，2倍单位矩阵

np.set_printoptions(threshold=np.inf) 
# 生成数据集合 X1
def generate_X1(N):
    # 随机选择标签，均匀分布于0, 1, 2（对应于 w1, w2, w3）
    labels = np.random.choice([0, 1, 2], size=N)  
    X1 = np.zeros((N, 2))  # 初始化数据集合 X1
    for i in range(N):
        # 根据随机标签生成二维随机向量
        if labels[i] == 0:
            X1[i] = np.random.multivariate_normal(mu_1, cov)
        elif labels[i] == 1:
            X1[i] = np.random.multivariate_normal(mu_2, cov)
        else:
            X1[i] = np.random.multivariate_normal(mu_3, cov)
    return X1, labels  # 返回生成的数据和标签

# 生成数据集合 X2
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
# predictions_X1_likelihood = classify(X1, method='likelihood')  # X1 的似然率分类
X2, labels2 = generate_X2(N)

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

# 创建一个 2 行 3 列的子图
fig = plt.figure(figsize=(18, 12))

# 创建网格
x_min_X1, x_max_X1 = X1[:, 0].min() - 1, X1[:, 0].max() + 1# X1 数据集的最小和最大值
y_min_X1, y_max_X1 = X1[:, 1].min()-1, X1[:, 1].max()+1
x_min_X2, x_max_X2 = X2[:, 0].min()-1, X2[:, 0].max()+1  # X2 数据集的最小和最大值
y_min_X2, y_max_X2 = X2[:, 1].min()-1, X2[:, 1].max()+1

grid_x_X1 = np.linspace(x_min_X1, x_max_X1, 200)
grid_y_X1 = np.linspace(y_min_X1, y_max_X1, 200)

grid_x_X2 = np.linspace(x_min_X2, x_max_X2, 200)
grid_y_X2 = np.linspace(y_min_X2, y_max_X2, 200)

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