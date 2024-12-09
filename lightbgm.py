import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def load_data(train_feature_path, train_label_path, test_feature_path):
    """
    加载训练和测试数据
    """
    # 加载训练特征
    with open(train_feature_path, 'rb') as f:
        train_features = pickle.load(f)
    
    # 加载训练标签
    train_labels = np.load(train_label_path)
    
    # 加载测试特征
    with open(test_feature_path, 'rb') as f:
        test_features = pickle.load(f)
    
    return train_features, train_labels, test_features

def apply_truncated_svd(train_X, test_X, n_components=5000):
    """
    使用 Truncated SVD 进行降维
    """
    print(f"应用 Truncated SVD 降维到 {n_components} 维...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(train_X)
    
    train_X_reduced = svd.transform(train_X)
    test_X_reduced = svd.transform(test_X)
    
    print(f"降维后训练特征形状: {train_X_reduced.shape}")
    print(f"降维后测试特征形状: {test_X_reduced.shape}")
    
    return train_X_reduced, test_X_reduced, svd

def train_logistic_regression(X_train, y_train):
    """
    训练逻辑回归模型，并进行超参数调优
    """
    print("训练逻辑回归模型...")
    lr = LogisticRegression(
        multi_class='multinomial',
        solver='saga',
        max_iter=1000,
        n_jobs=-1,
        random_state=42
    )
    
    # 定义参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['saga']
    }
    
    # 使用 Stratified K-Fold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    best_lr = grid_search.best_estimator_
    
    return best_lr

def train_linear_svm(X_train, y_train):
    """
    训练线性支持向量机模型，并进行超参数调优
    """
    print("训练线性支持向量机模型...")
    svm = LinearSVC(
        multi_class='ovr',
        max_iter=10000,
        random_state=42,
        dual=False,
        tol=1e-4
    )
    
    # 定义参数网格
    param_grid = {
        'C': [0.01, 0.1, 1, 10]
    }
    
    # 使用 Stratified K-Fold 进行交叉验证
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        scoring='accuracy',
        cv=skf,
        verbose=2,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    best_svm = grid_search.best_estimator_
    
    return best_svm

def plot_confusion_matrix(y_true, y_pred, num_classes, title='Confusion Matrix'):
    """
    绘制混淆矩阵
    """
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()

def main():
    # 数据路径
    train_feature_path = './data/train_feature.pkl'
    train_label_path = './data/train_labels.npy'
    test_feature_path = './data/test_feature.pkl'
    
    # 检查数据文件是否存在
    if not os.path.exists(train_feature_path):
        print(f"训练特征文件 {train_feature_path} 不存在！")
        return
    if not os.path.exists(train_label_path):
        print(f"训练标签文件 {train_label_path} 不存在！")
        return
    if not os.path.exists(test_feature_path):
        print(f"测试特征文件 {test_feature_path} 不存在！")
        return
    
    # 加载数据
    print("加载数据...")
    train_X, train_y, test_X = load_data(train_feature_path, train_label_path, test_feature_path)
    print(f"训练特征形状: {train_X.shape}")
    print(f"训练标签形状: {train_y.shape}")
    print(f"测试特征形状: {test_X.shape}")
    
    # 划分训练集和验证集
    print("划分训练集和验证集...")
    X_train, X_valid, y_train, y_valid = train_test_split(
        train_X, train_y, test_size=0.2, random_state=42, stratify=train_y
    )
    print(f"训练集形状: {X_train.shape}, {y_train.shape}")
    print(f"验证集形状: {X_valid.shape}, {y_valid.shape}")
    
    # 应用 Truncated SVD 进行降维
    X_train_reduced, X_valid_reduced, svd = apply_truncated_svd(X_train, X_valid, n_components=5000)
    test_X_reduced = svd.transform(test_X)
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_reduced = scaler.fit_transform(X_train_reduced)
    X_valid_reduced = scaler.transform(X_valid_reduced)
    test_X_reduced = scaler.transform(test_X_reduced)
    
    # 训练逻辑回归模型
    best_lr = train_logistic_regression(X_train_reduced, y_train)
    
    # 训练线性支持向量机模型
    best_svm = train_linear_svm(X_train_reduced, y_train)
    
    # 在验证集上评估逻辑回归模型
    print("\n评估逻辑回归模型...")
    y_valid_pred_lr = best_lr.predict(X_valid_reduced)
    valid_accuracy_lr = accuracy_score(y_valid, y_valid_pred_lr)
    print(f"逻辑回归验证集准确率: {valid_accuracy_lr:.4f}")
    print("分类报告（逻辑回归）:")
    print(classification_report(y_valid, y_valid_pred_lr))
    
    # 绘制逻辑回归的混淆矩阵
    plot_confusion_matrix(y_valid, y_valid_pred_lr, num_classes=20, title='Logistic Regression Confusion Matrix')
    
    # 在验证集上评估线性支持向量机模型
    print("\n评估线性支持向量机模型...")
    y_valid_pred_svm = best_svm.predict(X_valid_reduced)
    valid_accuracy_svm = accuracy_score(y_valid, y_valid_pred_svm)
    print(f"线性支持向量机验证集准确率: {valid_accuracy_svm:.4f}")
    print("分类报告（线性支持向量机）:")
    print(classification_report(y_valid, y_valid_pred_svm))
    
    # 绘制线性支持向量机的混淆矩阵
    plot_confusion_matrix(y_valid, y_valid_pred_svm, num_classes=20, title='Linear SVM Confusion Matrix')
    
    # 选择性能更好的模型进行测试集预测
    if valid_accuracy_lr >= valid_accuracy_svm:
        best_model = best_lr
        print("\n选择逻辑回归作为最终模型进行测试集预测。")
    else:
        best_model = best_svm
        print("\n选择线性支持向量机作为最终模型进行测试集预测。")
    
    # 对测试集进行预测
    print("对测试集进行预测...")
    test_pred = best_model.predict(test_X_reduced)
    
    # 创建提交文件
    print("创建提交文件...")
    submission = pd.DataFrame({
        'ID': np.arange(0, len(test_pred)),
        'Category': test_pred
    })
    
    # 确保输出目录存在
    output_dir = './data'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存为 CSV
    submission_path = os.path.join(output_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"提交文件已保存到 {submission_path}")
    
    # 保存模型（可选）
    print("保存模型...")
    joblib.dump(best_lr, os.path.join(output_dir, 'best_logistic_regression_model.joblib'))
    joblib.dump(best_svm, os.path.join(output_dir, 'best_linear_svm_model.joblib'))
    print("模型已保存。")

if __name__ == "__main__":
    main()
