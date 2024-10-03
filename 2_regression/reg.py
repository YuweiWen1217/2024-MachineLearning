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

#标准化数据，对训练集标准化后再将规则用于测试集
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 在X_train和X_test的末尾添加全为1的一列
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # 添加全1列到训练集
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))      # 添加全1列到测试集


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
            last_mse = 1000
            epsilon=1e-4
            isComplete = False
            for _ in range(self.iterations):
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
                    if abs(mse - last_mse) < epsilon:
                        isComplete = True
                        break
                    else:
                        last_mse = mse
                    # 计算损失函数相对于权重的梯度
                    gradient = xi.T.dot(error)  # 计算梯度，xi.T 为特征的转置
                    # 更新权重参数，使用学习率控制更新幅度
                    self.theta -= self.learning_rate * gradient
                    self.mse_history.append(mse)
                if isComplete:
                    break
        else:
            raise ValueError("Invalid method. Choose 'batch' or 'stochastic'.")
    
    def predict(self, X):
        # 根据输入特征X进行预测
        predictions = X.dot(self.theta)  # 计算预测值
        #return np.round(predictions).astype(int)  # 四舍五入并转换为整数
        return predictions


# 定义不同的学习率
learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

print('批量梯度下降')
plt.figure(figsize=(10, 6))
best_lr = -1
best_test_mse = 100
for lr in learning_rates:
    # 初始化模型
    model = LinearRegression(learning_rate=lr, iterations=1000, method='batch')
    model.fit(X_train, y_train)
    # 绘制当前学习率的MSE历史
    plt.plot(model.mse_history, label=f'LR: {lr}')
    mse_train = np.mean((model.predict(X_train) - y_train) ** 2)
    mse_test = np.mean((model.predict(X_test) - y_test) ** 2)
    print(f'学习率 {lr}  | 训练集mse  {mse_train:.4f}  | 测试集mse: {mse_test:.4f}')
    if mse_test < best_test_mse:
        best_test_mse = mse_test
        best_lr = lr
print(f'最佳学习率为：{best_lr} | 对应测试集mse为: {best_test_mse:.4f}')

plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('MSE Curve of Batch Gradient Descent under Different LRs')
plt.legend()
plt.grid()
plt.show()


print('随机梯度下降')
# 定义不同的学习率
learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
best_lr = -1
best_test_mse = 100
# 绘制每个学习率的训练曲线
plt.figure(figsize=(10, 6))
for lr in learning_rates:
    # 初始化模型
    model = LinearRegression(learning_rate=lr, iterations=1000, method='stochastic')
    model.fit(X_train, y_train)
    # 绘制当前学习率的MSE历史
    plt.plot(model.mse_history, label=f'LR: {lr}')
    mse_train = np.mean((model.predict(X_train) - y_train) ** 2)
    mse_test = np.mean((model.predict(X_test) - y_test) ** 2)
    print(f'学习率 {lr}  | 训练集mse  {mse_train:.4f}  | 测试集mse: {mse_test:.4f}')
    if mse_test < best_test_mse:
        best_test_mse = mse_test
        best_lr = lr
print(f'最佳学习率为：{best_lr} | 对应测试集mse为: {best_test_mse:.4f}')
# 绘制图形
plt.xlabel('Iterations')
plt.ylabel('MSE')
plt.title('MSE Curve of Stochastic Gradient Descent under Different LRs')
plt.legend()
plt.grid()
plt.show()



# 岭回归解析解
def ridge_regression(X, y, alpha):
    m, n = X.shape
    I = np.eye(n)  # 单位矩阵
    theta = np.linalg.inv(X.T.dot(X) + alpha * I).dot(X.T).dot(y)
    return theta

# 预测和计算误差
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# 训练岭回归模型
alpha = 0.1 # 正则化参数
theta_ridge = ridge_regression(X_train, y_train, alpha)
y_train_pred = X_train.dot(theta_ridge)
y_test_pred = X_test.dot(theta_ridge)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
print(f"岭回归(alpha = 0.1)训练误差: {train_mse:.4f}, 测试误差: {test_mse:.4f}")