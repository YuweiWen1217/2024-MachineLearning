import numpy as np
import matplotlib.pyplot as plt

# 加载semeion.data数据集
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    # 前256列是特征，后10列是one-hot编码的标签
    X = data[:, :256]
    # 将one-hot编码转换为单一的数字标签
    y = np.argmax(data[:, 256:], axis=1)
    return X, y

# 绘制手写数字
def plot_digit_image(pixel_data, label):
    # 将256个像素值转换成16x16的图像
    image = pixel_data.reshape(16, 16)
    
    # 创建图像
    plt.imshow(image, cmap='gray')
    plt.title(f"Handwritten Digit: {label}")
    plt.axis('off')  # 隐藏坐标轴
    # 显示图像
    plt.show()

# 加载数据
file_path = 'semeion.data'  # 替换为你文件的实际路径
X, y = load_semeion_data(file_path)

# 选择一行数据作为示例（例如第0行）
sample_index = 462  # 你可以替换为其他行的索引
sample_pixel_data = X[sample_index]
sample_label = y[sample_index]
# 绘制还原后的手写数字图像
plot_digit_image(sample_pixel_data, sample_label)
