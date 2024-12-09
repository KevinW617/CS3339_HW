import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import issparse, csr_matrix
import warnings
import os

# 忽略警告信息
warnings.filterwarnings('ignore')

def load_data(train_feature_path, train_label_path, test_feature_path):
    """
    加载训练特征、训练标签和测试特征。
    """
    print("加载数据...")
    with open(train_feature_path, 'rb') as f:
        X_train = pickle.load(f)
    
    y_train = np.load(train_label_path)
    
    with open(test_feature_path, 'rb') as f:
        X_test = pickle.load(f)
    
    print(f"训练集特征维度: {X_train.shape}")
    print(f"训练集标签维度: {y_train.shape}")
    print(f"测试集特征维度: {X_test.shape}")
    
    return X_train, y_train, X_test

def preprocess_data(X_train, X_test):
    """
    处理稀疏矩阵或转换数据。
    """
    print("预处理数据...")
    if issparse(X_train):
        X_train_processed = X_train
    else:
        X_train_processed = csr_matrix(X_train)
    
    if issparse(X_test):
        X_test_processed = X_test
    else:
        X_test_processed = csr_matrix(X_test)
    
    return X_train_processed, X_test_processed

def train_model(X_train, y_train):
    """
    训练SVM模型并评估性能。
    """
    print("训练SVM模型...")
    model = LinearSVC(
        C=1.0,               # 正则化参数
        max_iter=1000,       # 最大迭代次数
        random_state=42,     # 随机种子
        class_weight='balanced',  # 处理类别不平衡
        verbose=1            # 显示训练进度
    )
    
    model.fit(X_train, y_train)
    
    # 计算训练集上的准确率
    train_predictions = model.predict(X_train)
    correct_predictions = np.sum(train_predictions == y_train)
    total_samples = len(y_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    print(f"\n训练集准确率: {correct_predictions}/{total_samples} = {train_accuracy:.4f}")
    
    # 打印每个类别的详细结果
    print("\n各类别的详细结果:")
    for class_label in range(20):
        class_mask = y_train == class_label
        class_total = np.sum(class_mask)
        class_correct = np.sum((train_predictions == y_train) & class_mask)
        class_accuracy = class_correct / class_total if class_total > 0 else 0
        print(f"类别 {class_label}: {class_correct}/{class_total} = {class_accuracy:.4f}")
    
    print("\n分类报告:")
    print(classification_report(y_train, train_predictions))
    
    return model

def save_predictions(predictions, output_path):
    """
    将预测结果保存为CSV文件。
    """
    submission_df = pd.DataFrame({
        'ID': range(len(predictions)),
        'label': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至 {output_path}")

def main():
    """
    主函数，执行完整的训练和预测流程。
    """
    data_dir = './data'
    
    # 设置文件路径
    train_feature_path = os.path.join(data_dir, 'train_feature.pkl')
    train_label_path = os.path.join(data_dir, 'train_labels.npy')
    test_feature_path = os.path.join(data_dir, 'test_feature.pkl')
    
    # 检查文件是否存在
    for path in [train_feature_path, train_label_path, test_feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")
    
    # 加载数据
    X_train, y_train, X_test = load_data(train_feature_path, train_label_path, test_feature_path)
    
    # 预处理数据
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # 训练模型
    model = train_model(X_train, y_train)
    
    # 预测测试集
    print("\n预测测试集...")
    test_predictions = model.predict(X_test)
    
    # 保存预测结果
    output_path = os.path.join(data_dir, 'test_predictions_svm.csv')
    save_predictions(test_predictions, output_path)
    
    print("\n========== 模型训练与预测完成 ==========")

if __name__ == "__main__":
    main()