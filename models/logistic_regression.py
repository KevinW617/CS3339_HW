import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
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
    with open(train_feature_path, 'rb') as f:
        X_train = pickle.load(f)
    
    y_train = np.load(train_label_path)
    
    with open(test_feature_path, 'rb') as f:
        X_test = pickle.load(f)
    
    return X_train, y_train, X_test

def preprocess_data(X_train, X_test):
    """
    处理稀疏矩阵或转换数据。
    """
    if issparse(X_train):
        X_train_processed = X_train
    else:
        X_train_processed = csr_matrix(X_train)
    
    if issparse(X_test):
        X_test_processed = X_test
    else:
        X_test_processed = csr_matrix(X_test)
    
    return X_train_processed, X_test_processed

def build_model():
    """
    构建逻辑回归模型，并设置超参数来增加过拟合风险。
    """
    model = LogisticRegression(
        penalty='l2',          # 正则化类型
        C=15000.0,           # 更小的正则化强度，减少正则化
        solver='lbfgs',        # 优化算法
        max_iter=100000000,    # 增加最大迭代次数，避免过早停止
        random_state=42,       # 随机种子
        multi_class='auto',    # 多分类策略
        class_weight='balanced',     # 不使用类别加权
        verbose=1,             # 输出更多训练信息
        tol=1e-10              # 设置更低的容忍度
    )
    return model

def train_and_evaluate(X_train, y_train):
    """训练模型并评估性能"""
    model = build_model()

    print("开始训练逻辑回归模型...")
    model.fit(X_train, y_train)
    print("训练完成。")

    # 基本性能评估
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
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
    将预测结果保存为CSV文件，ID从0开始。
    """
    submission_df = pd.DataFrame({
        'ID': np.arange(0, len(predictions)),
        'Category': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至 {output_path}")

def main():
    """主函数，执行数据加载、预处理、模型训练、评估和预测保存。"""
    data_dir = '../data'

    train_feature_path = os.path.join(data_dir, 'train_feature.pkl')
    train_label_path = os.path.join(data_dir, 'train_labels.npy')
    test_feature_path = os.path.join(data_dir, 'test_feature.pkl')

    for path in [train_feature_path, train_label_path, test_feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")
        else:
            print(f"找到文件: {path}")

    print("\n加载数据...")
    X_train, y_train, X_test = load_data(train_feature_path, train_label_path, test_feature_path)
    print("数据加载完成。")

    print("\n预处理数据...")
    X_train, X_test = preprocess_data(X_train, X_test)
    print("数据预处理完成。")

    print("\n训练并评估模型...")
    best_model, evaluation_metrics = train_and_evaluate(X_train, y_train)

    print("\n对测试集进行预测...")
    test_predictions = best_model.predict(X_test)

    output_path = os.path.join(data_dir, 'test_predictions_logistic_regression.csv')
    save_predictions(test_predictions, output_path)

    print("\n========== 模型训练与预测完成 ==========")

if __name__ == "__main__":
    main()
