import numpy as np

# 创建一个包含6个元素的数组
arr = np.array([1, 2, 3, 4, 5, 6])

# 将一维数组重塑为二维数组，自动计算行数
reshaped = arr.reshape(-1)

print("原数组:", arr)
print("重新调整形状数组:", reshaped)
