import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


MAX_NUM = 1e3

# method
def singleLinkage(cluster_distances):
    # 单链接聚类：取两个聚类之间的最小距离
    return np.min(cluster_distances, axis=0)

def completeLinkage(cluster_distances):
    # 完全链接聚类：取两个聚类之间的最大距离
    return np.max(cluster_distances, axis=0)

def averageLinkage(cluster_distances, m, n):
    new_distances = (m * cluster_distances[0] + n * cluster_distances[1]) / (m + n)
    return new_distances
    


class AgglomerativeClustering:
    def __init__(self):
        # 初始化类
        self.steps = []  # 用于记录聚类过程中的合并步骤，每次合并记录两个簇的代表点索引。

    def fit(self, datas, method):
        """
        执行层次聚类的核心方法。

        :param datas: 数据集，形状为 (n_samples, n_features) 的 numpy 数组。
        :param method: 函数对象，用于计算两个簇之间的距离（如 single-linkage, complete-linkage 等）。
        """
        self.dataCnt = datas.shape[0]  # 数据点的数量
        allDist = np.zeros((self.dataCnt, self.dataCnt))  # 存储所有数据点之间的距离矩阵

        # 计算每对点之间的欧式距离平方，填充到距离矩阵中
        for i in range(self.dataCnt):
            for j in range(i):
                allDist[i][j] = allDist[j][i] = np.sum((datas[i] - datas[j]) ** 2)
        setList, clusterCount = [[i] for i in range(self.dataCnt)], self.dataCnt  # 初始化每个点为独立簇
        print("calculate distance finish!")  # 打印距离计算完成的信息

        # 初始化簇间距离矩阵，初始值为无穷大
        clusterDist = np.zeros((self.dataCnt, self.dataCnt)) + MAX_NUM
        for i in range(clusterCount):
            for j in range(i + 1, clusterCount):
                clusterDist[i][j] = clusterDist[j][i] = allDist[i][j]  # 初始的簇间距离等于数据点之间的距离
        print("calculate cluster distance finish!")  # 打印簇间距离初始化完成的信息

        # 聚类过程，直到剩余的簇数量为 4（可以根据需要调整停止条件）
        while clusterCount != 4:
            # 找到当前簇间距离矩阵中距离最小的两个簇
            res = np.argmin(clusterDist)  # 获取最小值的索引
            dest, src = int(res / clusterCount), res % clusterCount  # 将索引转为矩阵中的行、列（簇编号）
            self.steps.append((setList[dest][0], setList[src][0]))  # 记录本次合并的两个簇

            # 根据 `method` 函数更新合并后的簇与其他簇的距离
            modify = method(clusterDist[[dest, src]])  
            clusterDist[dest] = modify  # 更新合并后簇的行
            clusterDist[:, dest] = modify  # 更新合并后簇的列
            clusterDist = np.delete(clusterDist, src, axis=0)  # 删除被合并簇的行
            clusterDist = np.delete(clusterDist, src, axis=1)  # 删除被合并簇的列
            clusterDist[dest][dest] = MAX_NUM  # 设置自身距离为无穷大

            # 更新簇集合，将被合并簇的点加入合并后的簇
            setList[dest] = setList[dest] + setList[src]
            del setList[src]  # 删除被合并的簇
            clusterCount -= 1  # 更新簇的总数

            # 打印进度信息，每完成一定比例的合并时输出
            if (self.dataCnt - clusterCount) % (self.dataCnt / 20) == 0:
                print(clusterCount, " clusters left.")
        print("cluster finish !")  # 打印聚类完成的信息

    def label(self, k):
        """
        根据合并记录生成簇标签。

        :param k: 指定最终聚类结果中的簇数量。
        :return: 簇标签列表，每个数据点对应一个簇标签。
        """
        root = list(range(self.dataCnt))  # 初始化每个点为自身的根节点

        def find_root(n):
            """
            找到节点 n 的根节点，并执行路径压缩。
            """
            if root[root[n]] == root[n]:
                return root[n]  # 当前节点的父节点即为根节点
            root[n] = find_root(root[n])  # 路径压缩，将节点直接连接到根节点
            return root[n]

        # 根据记录的合并步骤更新根节点
        for i in range(self.dataCnt - k):  # 仅执行 (dataCnt - k) 次合并，保留 k 个簇
            src, dest = self.steps[i]
            root[find_root(dest)] = find_root(src)  # 将一个簇的根节点合并到另一个簇

        # 分配簇标签
        cluster, clusterNum = [0 for _ in range(self.dataCnt)], 0
        for i in range(self.dataCnt):
            if i == root[i]:  # 当前节点是根节点，分配新的簇编号
                clusterNum += 1
                cluster[i] = clusterNum
        for i in range(self.dataCnt):
            if i != root[i]:  # 非根节点继承其根节点的簇编号
                cluster[i] = cluster[find_root(i)]
        return cluster

    


def create_data(centers,num=100,std=0.7):
    '''
    生成用于聚类的数据集
    :param centers: 聚类的中心点组成的数组。如果中心点是二维的，则产生的每个样本都是二维的。
    :param num: 样本数
    :param std: 每个簇中样本的标准差
    :return: 用于聚类的数据集。是一个元组，第一个元素为样本集，第二个元素为样本集的真实簇分类标记
    '''
    X, labels_true = make_blobs(n_samples=num, centers=centers, cluster_std=std)
    return  X,labels_true


def plot_data(*data):
    '''
    绘制用于聚类的数据集
    :param data: 可变参数。它是一个元组。元组元素依次为：第一个元素为样本集，第二个元素为样本集的真实簇分类标记，第三个元素为预测分类标记
    :return: None
    '''
    X,labels_true,labels_predict=data
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    colors='rgbyckm' # 每个簇的样本标记不同的颜色
    markers='o^sP*DX'
    for i in range(len(labels_true)):
        predict=labels_predict[i]
        ax.scatter(X[i,0],X[i,1],label="cluster %d"%labels_true[i],
        color=colors[predict%len(colors)],marker=markers[labels_true[i]%len(markers)],alpha=0.5)
    plt.show()





centers=[[1,1,1], [1,3,3], [3,6,5], [2,6,8]]# 用于产生聚类的中心点, 聚类中心的维度代表产生样本的维度
X, labels_true= create_data(centers, 2000, 0.5) # 产生用于聚类的数据集，聚类中心点的个数代表类别数
# 绘制生成的三维数据点
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 按真实标签绘制不同颜色的点
for label in set(labels_true):
    cluster_points = X[labels_true == label]
    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2], label=f'Cluster {label}', s=10)

ax.set_title("3D Scatter Plot of Generated Data")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()




METHOD_APPLY = [singleLinkage, completeLinkage, averageLinkage]

k_values = [2, 3, 4, 5, 6] 
ari_single_k = []
ari_complete_k = []
ari_average_k = []

for k in k_values:
    centers = np.random.rand(k, 3) * 10  # 随机生成k个聚类中心
    X, labels_true = create_data(centers, num=k*500)
    for method in METHOD_APPLY:
        model = AgglomerativeClustering()
        model.fit(X, method)
        labels_predict = model.label(k)
        ari = adjusted_rand_score(labels_true, labels_predict)
        if method == singleLinkage:
            ari_single_k.append(ari)
        elif method == completeLinkage:
            ari_complete_k.append(ari)
        elif method == averageLinkage:
            ari_average_k.append(ari)

# 绘制不同聚类数下的ARI变化
plt.figure(figsize=(10, 6))
plt.plot(k_values, ari_single_k, label='Single-Linkage', marker='o')
plt.plot(k_values, ari_complete_k, label='Complete-Linkage', marker='s')
plt.plot(k_values, ari_average_k, label='Average-Linkage', marker='^')

plt.xlabel('Number of Clusters (k)', fontsize=12)
plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)
plt.title('Performance Comparison of Different Clustering Methods', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
    