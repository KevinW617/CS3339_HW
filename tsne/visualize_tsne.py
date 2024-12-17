import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import os
from scipy.sparse import issparse

def load_data(train_feature_path, train_label_path):
    """加载训练数据和标签"""
    print("加载数据...")
    with open(train_feature_path, 'rb') as f:
        X_train = pickle.load(f)
    
    y_train = np.load(train_label_path)
    
    if issparse(X_train):
        X_train = X_train.toarray()
    
    print(f"数据维度: {X_train.shape}")
    print(f"标签维度: {y_train.shape}")
    return X_train, y_train

def visualize_tsne(X, y):
    """使用t-SNE进行降维并可视化 (perplexity=80)"""
    print("执行t-SNE降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=80,  # 使用固定的perplexity值
        n_iter=1000,
        random_state=42,
        verbose=1
    )
    
    # 执行t-SNE降维
    X_tsne = tsne.fit_transform(X)
    
    # 创建图形
    plt.figure(figsize=(12, 8))
    
    # 使用散点图可视化
    scatter = plt.scatter(
        X_tsne[:, 0],
        X_tsne[:, 1],
        c=y,
        cmap='tab20',
        alpha=0.6
    )
    
    # 添加颜色条
    plt.colorbar(scatter)
    
    # 设置标题和标签
    plt.title('t-SNE Visualization of Text Classification Data (perplexity=80)')
    plt.xlabel('t-SNE dimension 1')
    plt.ylabel('t-SNE dimension 2')
    
    # 保存图形
    plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
    print("可视化结果已保存为 tsne_visualization.png")
    
    # 创建每个类别单独的可视化
    plt.figure(figsize=(20, 15))
    for i in range(20):  # 20个类别
        plt.subplot(4, 5, i+1)  # 4行5列的子图
        mask = y == i
        plt.scatter(
            X_tsne[mask, 0],
            X_tsne[mask, 1],
            c='b',
            alpha=0.6,
            label=f'Class {i}'
        )
        plt.scatter(
            X_tsne[~mask, 0],
            X_tsne[~mask, 1],
            c='gray',
            alpha=0.1
        )
        plt.title(f'Class {i}')
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.savefig('tsne_visualization_by_class.png', dpi=300, bbox_inches='tight')
    print("分类别可视化结果已保存为 tsne_visualization_by_class.png")

def visualize_tsne_multiple_perplexities(X, y, perplexities=[5, 30, 50, 100, 150, 200, 300, 500]):
    """使用不同的perplexity值进行t-SNE可视化"""
    n_plots = len(perplexities)
    n_cols = 4  # 每行4个图
    n_rows = (n_plots + n_cols - 1) // n_cols  # 计算需要的行数
    
    plt.figure(figsize=(20, 5 * n_rows))  # 调整图形大小以适应多行
    
    for idx, perp in enumerate(perplexities):
        print(f"执行t-SNE降维 (perplexity={perp})...")
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            n_iter=1000,
            random_state=42,
            verbose=1
        )
        
        X_tsne = tsne.fit_transform(X)
        
        plt.subplot(n_rows, n_cols, idx+1)  # 使用动态计算的行数
        scatter = plt.scatter(
            X_tsne[:, 0],
            X_tsne[:, 1],
            c=y,
            cmap='tab20',
            alpha=0.6
        )
        
        plt.title(f'Perplexity = {perp}')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
    
    plt.tight_layout()
    plt.savefig('tsne_perplexity_comparison.png', dpi=300, bbox_inches='tight')
    print("不同perplexity值的可视化结果已保存为 tsne_perplexity_comparison.png")

def main():
    data_dir = './data'
    train_feature_path = os.path.join(data_dir, 'train_feature.pkl')
    train_label_path = os.path.join(data_dir, 'train_labels.npy')
    
    # 检查文件是否存在
    for path in [train_feature_path, train_label_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")
    
    # 加载数据
    X_train, y_train = load_data(train_feature_path, train_label_path)
    
    # 可视化不同perplexity值的效果
    visualize_tsne_multiple_perplexities(X_train, y_train)
    
    # 使用默认perplexity进行标准可视化
    visualize_tsne(X_train, y_train)

if __name__ == "__main__":
    main() 