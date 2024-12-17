import numpy as np
import pandas as pd
import pickle
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import issparse, csr_matrix
import warnings
import os
from model_evaluation import evaluate_model

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
    """训练SVM模型并进行评估"""
    print("训练SVM模型...")
    model = LinearSVC(
        C=0.5,
        max_iter=10000,
        random_state=42,
        class_weight='balanced',
        verbose=1
    )
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 基本性能评估
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    print(f"\n训练集准确率: {train_accuracy:.4f}")
    
    # 综合模型评估
    print("\n执行综合模型评估...")
    evaluation_metrics = evaluate_model(model, X_train, y_train)
    
    # 打印评估结果
    print("\n=== 偏差-方差分析 ===")
    print(f"偏差: {evaluation_metrics['bias_variance']['bias']:.4f}")
    print(f"方差: {evaluation_metrics['bias_variance']['variance']:.4f}")
    print(f"总误差: {evaluation_metrics['bias_variance']['total_error']:.4f}")
    
    print("\n=== 交叉验证结果 ===")
    print(f"平均错误率: {evaluation_metrics['cross_validation']['mean_cv_score']:.4f}")
    print(f"标准差: {evaluation_metrics['cross_validation']['std_cv_score']:.4f}")
    
    print("\n=== 信息准则 ===")
    print(f"AIC: {evaluation_metrics['information_criteria']['aic']:.2f}")
    print(f"BIC: {evaluation_metrics['information_criteria']['bic']:.2f}")
    
    return model, evaluation_metrics

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
    data_dir = '../data'
    
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
    model, evaluation_metrics = train_model(X_train, y_train)
    
    # 预测测试集
    print("\n预测测试集...")
    test_predictions = model.predict(X_test)
    
    # 保存预测结果
    output_path = os.path.join(data_dir, 'test_predictions_svm.csv')
    save_predictions(test_predictions, output_path)
    
    print("\n========== 模型训练与预测完成 ==========")

if __name__ == "__main__":
    main()