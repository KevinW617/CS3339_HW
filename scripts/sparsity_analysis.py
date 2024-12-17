import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.sparse import load_npz

# Load training data
# Assuming `train_feature.pkl` is a pickled sparse matrix
with open('../data/train_feature.pkl', 'rb') as f:
    X_train = pickle.load(f)

# Load label data
y_train = np.load('../data/train_labels.npy')

# Check the type and shape of the loaded data
print(f"X_train type: {type(X_train)}")
print(f"y_train type: {type(y_train)}")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Compute sparsity
non_zero_elements = X_train.count_nonzero()  # Count non-zero elements
total_elements = X_train.shape[0] * X_train.shape[1]  # Total number of elements
sparsity = 1 - non_zero_elements / total_elements  # Sparsity
print(f"Sparsity: {sparsity:.4f}")

# Compute the number of non-zero features per sample
non_zero_per_sample = np.array(X_train.getnnz(axis=1))

# Compute the number of non-zero occurrences per feature
non_zero_per_feature = np.array(X_train.getnnz(axis=0))

# Statistics for non-zero features per sample
avg_non_zero = non_zero_per_sample.mean()  # Average number of non-zero features per sample
max_non_zero = non_zero_per_sample.max()  # Maximum number of non-zero features per sample
min_non_zero = non_zero_per_sample.min()  # Minimum number of non-zero features per sample

# Statistics for non-zero occurrences per feature
avg_non_zero_feature = non_zero_per_feature.mean()  # Average number of non-zero occurrences per feature
max_non_zero_feature = non_zero_per_feature.max()  # Maximum number of non-zero occurrences per feature
min_non_zero_feature = non_zero_per_feature.min()  # Minimum number of non-zero occurrences per feature

# Output the statistics
print(f"Average number of non-zero features per sample: {avg_non_zero:.2f}")
print(f"Maximum number of non-zero features per sample: {max_non_zero}")
print(f"Minimum number of non-zero features per sample: {min_non_zero}")

print(f"Average number of non-zero occurrences per feature: {avg_non_zero_feature:.2f}")
print(f"Maximum number of non-zero occurrences per feature: {max_non_zero_feature}")
print(f"Minimum number of non-zero occurrences per feature: {min_non_zero_feature}")

# Visualize the distribution of non-zero features per sample
plt.figure(figsize=(10, 6))
plt.hist(non_zero_per_sample, bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Non-Zero Features per Sample')
plt.xlabel('Number of Non-Zero Features')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.show()

# Visualize the distribution of non-zero occurrences per feature
plt.figure(figsize=(10, 6))
plt.hist(non_zero_per_feature, bins=50, color='lightgreen', edgecolor='black')
plt.title('Distribution of Non-Zero Occurrences per Feature')
plt.xlabel('Number of Non-Zero Occurrences')
plt.ylabel('Number of Features')
plt.grid(True)
plt.show()

# Visualize the sparsity distribution for each sample
plt.figure(figsize=(10, 6))
plt.hist(1 - non_zero_per_sample / X_train.shape[1], bins=50, color='lightcoral', edgecolor='black')
plt.title('Sparsity Distribution for Samples')
plt.xlabel('Sparsity of Samples')
plt.ylabel('Number of Samples')
plt.grid(True)
plt.show()

# Visualize the sparsity distribution for each feature
plt.figure(figsize=(10, 6))
plt.hist(1 - non_zero_per_feature / X_train.shape[0], bins=50, color='lightblue', edgecolor='black')
plt.title('Sparsity Distribution for Features')
plt.xlabel('Sparsity of Features')
plt.ylabel('Number of Features')
plt.grid(True)
plt.show()
