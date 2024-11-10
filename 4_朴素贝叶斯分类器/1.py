import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.preprocessing import label_binarize

# 加载数据集（wine.data）
data = np.loadtxt('wine.data', delimiter=',')  # 假设wine.data是CSV格式的，按逗号分隔
X = data[:, 1:]  # 特征（去掉第一列）
y = data[:, 0].astype(int)  # 标签（第一列）

# 分层采样（自行实现）
def train_test_split_stratified(X, y, test_size=0.3):
    # 获取类别标签
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    for c in unique_classes:
        # 获取当前类别的所有样本
        class_indices = np.where(y == c)[0]
        # 按照比例划分训练集和测试集
        np.random.shuffle(class_indices)
        split_index = int(len(class_indices) * (1 - test_size))
        train_indices.extend(class_indices[:split_index])
        test_indices.extend(class_indices[split_index:])
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

# 使用分层采样划分数据
X_train, X_test, y_train, y_test = train_test_split_stratified(X, y, test_size=0.3)

# 实现朴素贝叶斯分类器
class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.class_probs = {}  # 每个类别的先验概率
        self.means = {}  # 每个类别中每个特征的均值
        self.stds = {}  # 每个类别中每个特征的标准差
        # 计算每个类别的先验概率
        for c in self.classes:
            class_samples = X[y == c]
            self.class_probs[c] = len(class_samples) / len(y)
            # 计算每个类别的均值和标准差
            self.means[c] = np.mean(class_samples, axis=0)
            self.stds[c] = np.std(class_samples, axis=0)
    def predict(self, X):
        predictions = []
        prob_matrix = []
        for x in X:
            class_probs = {}
            # 对每个类别计算后验概率
            for c in self.classes:
                # 计算类别的先验概率
                prior = np.log(self.class_probs[c]) 
                # 计算条件概率
                likelihood = -0.5 * np.sum(np.log(2 * np.pi * self.stds[c]**2)) - 0.5 * np.sum(((x - self.means[c])**2) / (self.stds[c]**2))
                # 总概率是先验和条件概率的乘积（取对数后为和）
                class_probs[c] = prior + likelihood
            prob_matrix.append(class_probs)
            predictions.append(max(class_probs, key=class_probs.get))
        prob_matrix = np.array([[class_probs[c] for c in self.classes] for class_probs in prob_matrix])
        return np.array(predictions), np.array(prob_matrix)

# 训练朴素贝叶斯分类器
model = NaiveBayes()
model.fit(X_train, y_train)
# 测试并计算准确率
y_pred, prob_matrix = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"分类准确率: {accuracy:.4f}")




# 计算混淆矩阵
conf_matrix = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)
for true, pred in zip(y_test, y_pred):
    conf_matrix[true - 1, pred - 1] += 1

print("混淆矩阵:")
print(conf_matrix)

# 计算精度、召回率和F1值
def precision_recall_f1(conf_matrix):
    precision = np.diag(conf_matrix) / np.sum(conf_matrix, axis=0)
    recall = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

precision, recall, f1 = precision_recall_f1(conf_matrix)

print("精度:", precision)
print("召回率:", recall)
print("F1值:", f1)


def roc(y_true, y_score, pos_label):
    """
    y_true: 真实标签，该类为1，不是该类为0
    y_score: 属于该类的分数（越高表示越有可能属于该类）
    pos_label: 1
    """
    # 统计正样本和负样本的个数
    num_positive_examples = (y_true == pos_label).sum()
    num_negtive_examples = len(y_true) - num_positive_examples
    tp, fp = 0, 0
    tpr, fpr = [], []
    score = max(y_score) + 1
    for i in np.flip(np.argsort(y_score)):
        if y_score[i] != score:
            fpr.append(fp / num_negtive_examples)
            tpr.append(tp / num_positive_examples)
            score = y_score[i]    
        if y_true[i] == pos_label:
            tp += 1
        else:
            fp += 1
    fpr.append(fp / num_negtive_examples)
    tpr.append(tp / num_positive_examples)
    return fpr, tpr

# 将 y_test 和 y_pred 二值化
y_test_bin = label_binarize(y_test, classes=np.unique(y))  # 实际标签

# 获取每个类别的 FPR 和 TPR
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}

# 假设类别标签是 0, 1, 2
for i in range(prob_matrix.shape[1]):  # 遍历每一列，对应每个类别
    # 对于每个类别，从 prob_matrix 中取出该类别的预测概率
    fpr, tpr = roc(y_test_bin[:, i], prob_matrix[:, i], pos_label=1)
    # 存储 FPR、TPR 和 AUC
    fpr_dict[i] = fpr
    tpr_dict[i] = tpr
    roc_auc_dict[i] = auc(fpr, tpr)

# 绘制 ROC 曲线
plt.figure()
colors = ['blue', 'green', 'red']
for i in range(prob_matrix.shape[1]):  # 绘制每个类别的 ROC 曲线
    plt.plot(fpr_dict[i], tpr_dict[i], color=colors[i], lw=2,
             label=f'Class {i + 1} (AUC = {roc_auc_dict[i]:.2f})')

plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()



