import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import rotate

# 加载数据
def load_semeion_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :256]  # 前256列是特征
    y = np.argmax(data[:, 256:], axis=1)  # 后10列是one-hot编码的标签
    return X, y

# 数据增强：旋转
def augment_data(image):
    rotated_left_up = rotate(image.reshape(16, 16), angle=20).flatten()
    rotated_left_down = rotate(image.reshape(16, 16), angle=-20).flatten()
    return rotated_left_up, rotated_left_down

# 显示图像
def show_images(original, rotated_left_up, rotated_left_down):
    plt.figure(figsize=(8, 6))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(original.reshape(16, 16), cmap='gray')
    plt.title('Original')
    plt.axis('off')

    # 左上旋转图像
    plt.subplot(1, 3, 2)
    plt.imshow(rotated_left_up.reshape(16, 16), cmap='gray')
    plt.title('Rotated Up (20°)')
    plt.axis('off')

    # 左下旋转图像
    plt.subplot(1, 3, 3)
    plt.imshow(rotated_left_down.reshape(16, 16), cmap='gray')
    plt.title('Rotated Down (-20°)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 主程序
file_path = 'semeion.data'
X, y = load_semeion_data(file_path)

# 输入指定行的索引（例如，索引为0的行）
index = 0
original_image = X[index]

# 增强数据
rotated_left_up, rotated_left_down = augment_data(original_image)

# 展示图像
show_images(original_image, rotated_left_up, rotated_left_down)
