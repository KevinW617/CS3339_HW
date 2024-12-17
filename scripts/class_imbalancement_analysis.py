import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: 加载数据
# 加载特征矩阵
with open('../data/train_feature.pkl', 'rb') as f:
    X_train = pickle.load(f)

# 加载标签数组
y_train = np.load('../data/train_labels.npy')

# Step 2: 分析类别分布
# 使用 pandas 的 value_counts() 来查看每个类别的样本数量
label_counts = pd.Series(y_train).value_counts()

# 打印类别分布
print("类别分布：")
print(label_counts)

# 计算类别不平衡的比例
imbalance_ratio = label_counts.max() / label_counts.min()
print(f"类别不平衡的比例: {imbalance_ratio:.2f}")

# Step 3: 可视化类别分布
# 绘制类别分布的条形图
plt.figure(figsize=(10, 6))
label_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Samples')
plt.title('Class Distribution in Training Set')
plt.xticks(rotation=90)  # 旋转类别标签，避免重叠
plt.show()

# Step 4: 如果需要进一步分析，可以使用其他可视化形式
# 绘制水平条形图
plt.figure(figsize=(10, 6))
label_counts.plot(kind='barh', color='skyblue')
plt.ylabel('Class')
plt.xlabel('Number of Samples')
plt.title('Class Distribution in Training Set (Horizontal View)')
plt.show()

# Step 5: 使用 numpy 进行类别计数（如果不想使用 pandas）
unique, counts = np.unique(y_train, return_counts=True)

print("\nNumPy统计：")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")

# Step 6: 进一步分析类别不平衡
# 计算每个类别的样本数量与最大样本数的比率
sample_ratios = counts / counts.max()
print("\n样本数量与最大类别的比率：")
for label, ratio in zip(unique, sample_ratios):
    print(f"Class {label}: {ratio:.2f}")

