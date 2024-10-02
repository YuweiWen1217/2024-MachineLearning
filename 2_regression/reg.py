import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据预处理
data = pd.read_csv('winequality-white.csv')
X = data.drop('quality', axis=1)
y = data['quality']
y = y.values.reshape(-1, 1)  

# 划分数据集，使用分层采样
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 标准化数据，对训练集标准化后再将规则用于测试集
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 在X_train和X_test的末尾添加全为1的一列
#X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 添加全1列到训练集
#X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))      # 添加全1列到测试集


class LinearRegression:
    def __init__(self, learning_rate=0.1, iterations=1000, method = None):
        # 初始化线性回归模型的参数
        self.learning_rate = learning_rate  # 学习率，控制每次更新的幅度
        self.iterations = iterations          # 迭代次数，模型训练的轮数
        self.theta = None                    # 权重参数，初始化为None
        self.method = method
        self.mse_history = []

    def fit(self, X, y):
        # 拟合模型，训练线性回归模型
        m, n = X.shape  # 获取训练数据的样本数和特征数
        self.theta = np.random.randn(n, 1)
        self.mse_history = []  # 用于记录每次迭代的均方误差

        # 批量梯度下降
        if self.method == 'batch':
            # 迭代训练模型
            for _ in range(self.iterations):
                predictions = X.dot(self.theta)  # 计算当前权重下的预测值
                errors = predictions - y  # 计算预测值与真实值之间的误差
                mse = np.mean(errors ** 2)  # 计算当前的均方误差
                self.mse_history.append(mse)  # 保存均方误差到历史记录中
                gradient = (1/m) * X.T.dot(errors)  # 计算损失函数相对于权重的梯度
                self.theta -= self.learning_rate * gradient  # 更新权重参数
        
        # 随机梯度下降
  
        elif self.method == 'stochastic':
            for _ in range(self.iterations):
                sum_mse = 0
                for i in range(m):
                    # 获取当前样本的特征
                    xi = X[i:i+1]  # 保持二维形状，以便于矩阵运算
                    # 获取当前样本的目标值
                    yi = y[i, 0]
                    # 计算当前权重下的预测值
                    prediction = xi.dot(self.theta)
                    # 计算预测值与真实值之间的误差
                    error = prediction - yi
                    # 计算当前的均方误差
                    mse = np.mean(error ** 2)
                    sum_mse += mse
                    # 计算损失函数相对于权重的梯度
                    gradient = xi.T.dot(error)  # 计算梯度，xi.T 为特征的转置
                    # 更新权重参数，使用学习率控制更新幅度
                    self.theta -= self.learning_rate * gradient
                self.mse_history.append(sum_mse/m)
        else:
            raise ValueError("Invalid method. Choose 'batch' or 'stochastic'.")
    
        

    def predict(self, X):
        # 根据输入特征X进行预测
        return X.dot(self.theta)  # 计算并返回预测值


# 批量梯度下降
batch_model = LinearRegression(method='batch', iterations=1000)
batch_model.fit(X_train, y_train)
batch_mse_train = np.mean((batch_model.predict(X_train) - y_train) ** 2)
batch_mse_test = np.mean((batch_model.predict(X_test) - y_test) ** 2)
print(batch_model.theta)


# 随机梯度下降
stochastic_model = LinearRegression(method='stochastic',iterations=1000, learning_rate=0.001)
stochastic_model.fit(X_train, y_train)
stochastic_mse_train = np.mean((stochastic_model.predict(X_train) - y_train) ** 2)
stochastic_mse_test = np.mean((stochastic_model.predict(X_test) - y_test) ** 2)
print(stochastic_model.theta)

# 绘制MSE收敛曲线
plt.plot(stochastic_model.mse_history, label='Stochastic Gradient Descent')
plt.plot(batch_model.mse_history, label='Batch Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Mean Squared Error')
plt.title('MSE Convergence')
plt.legend()
plt.show()

# 输出MSE
print(f'Batch Gradient Descent MSE - Train: {batch_mse_train}, Test: {batch_mse_test}')
print(f'Stochastic Gradient Descent MSE - Train: {stochastic_mse_train}, Test: {stochastic_mse_test}')
