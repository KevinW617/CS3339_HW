# svm_classifier_optimized.py

import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import issparse
from sklearn.ensemble import VotingClassifier
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

def load_data(train_feature_path, train_label_path, test_feature_path):
    """
    加载训练特征、训练标签和测试特征。
    """
    # 加载训练特征
    with open(train_feature_path, 'rb') as f:
        X_train = pickle.load(f)
    
    # 加载训练标签
    y_train = np.load(train_label_path)
    
    # 加载测试特征
    with open(test_feature_path, 'rb') as f:
        X_test = pickle.load(f)
    
    return X_train, y_train, X_test

def preprocess_data(X_train, X_test):
    """
    检查数据稀疏性，并根据需要进行转换。
    """
    if issparse(X_train):
        print("训练数据是稀疏矩阵")
    else:
        print("训练数据不是稀疏矩阵，转换为稀疏矩阵")
        from scipy.sparse import csr_matrix
        X_train = csr_matrix(X_train)
    
    if issparse(X_test):
        print("测试数据是稀疏矩阵")
    else:
        print("测试数据不是稀疏矩阵，转换为稀疏矩阵")
        from scipy.sparse import csr_matrix
        X_test = csr_matrix(X_test)
    
    return X_train, X_test

def build_pipeline(selected_model='SVM'):
    """
    构建包含特征选择、降维、标准化和分类器的管道。
    """
    # 特征选择：选择卡方检验得分最高的5000个特征
    feature_selection = SelectKBest(chi2, k=5000)
    
    # 降维：使用Truncated SVD将特征降至300维
    svd = TruncatedSVD(n_components=300, random_state=42)
    
    # 标准化：保持稀疏性，不减去均值
    scaler = StandardScaler(with_mean=False)
    
    # 定义分类器
    if selected_model == 'SVM':
        classifier = SVC(kernel='linear', probability=True, random_state=42)
    elif selected_model == 'LinearSVC':
        classifier = LinearSVC(random_state=42, max_iter=10000)
    elif selected_model == 'LogisticRegression':
        classifier = LogisticRegression(max_iter=1000, random_state=42)
    elif selected_model == 'RandomForest':
        classifier = RandomForestClassifier(random_state=42)
    elif selected_model == 'GradientBoosting':
        classifier = GradientBoostingClassifier(random_state=42)
    else:
        raise ValueError("Unsupported model type")
    
    # 构建管道
    pipeline = Pipeline([
        ('feature_selection', feature_selection),
        ('svd', svd),
        ('scaler', scaler),
        ('classifier', classifier)
    ])
    
    return pipeline

def define_param_grid(model_name):
    """
    定义不同模型的参数网格。
    """
    if model_name == 'SVM':
        param_grid = {
            'classifier__C': [0.1, 1, 10]
        }
    elif model_name == 'LinearSVC':
        param_grid = {
            'classifier__C': [0.1, 1, 10]
        }
    elif model_name == 'LogisticRegression':
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__penalty': ['l2']
        }
    elif model_name == 'RandomForest':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
    elif model_name == 'GradientBoosting':
        param_grid = {
            'classifier__n_estimators': [100, 200],
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__max_depth': [3, 5]
        }
    else:
        param_grid = {}
    return param_grid

def train_and_evaluate(X_train, y_train, model_name='SVM'):
    """
    训练模型并在验证集上评估性能。
    """
    # 构建管道
    pipeline = build_pipeline(selected_model=model_name)
    
    # 定义参数网格
    param_grid = define_param_grid(model_name)
    
    # 使用GridSearchCV进行超参数调优
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=2
    )
    
    # 拆分训练集和验证集
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    # 训练模型
    print(f"正在训练模型: {model_name} ...")
    grid_search.fit(X_train_part, y_train_part)
    print("训练完成。")
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    # 在验证集上评估模型
    val_predictions = grid_search.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"验证集准确率: {val_accuracy:.4f}")
    print("分类报告:")
    print(classification_report(y_val, val_predictions))
    
    return grid_search, val_accuracy

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
    # 定义数据路径
    train_feature_path = './data/train_feature.pkl'
    train_label_path = './data/train_labels.npy'
    test_feature_path = './data/test_feature.pkl'
    
    # 加载数据
    X_train, y_train, X_test = load_data(train_feature_path, train_label_path, test_feature_path)
    
    # 预处理数据
    X_train, X_test = preprocess_data(X_train, X_test)
    
    # 选择模型并训练
    # 您可以尝试 'SVM', 'LinearSVC', 'LogisticRegression', 'RandomForest', 'GradientBoosting'
    selected_models = ['SVM', 'LinearSVC', 'LogisticRegression', 'RandomForest', 'GradientBoosting']
    
    # 存储每个模型的最佳结果和验证集准确率
    best_models = {}
    model_accuracies = {}
    
    for model_name in selected_models:
        print(f"\n========== 训练模型: {model_name} ==========")
        grid_search, val_accuracy = train_and_evaluate(X_train, y_train, model_name=model_name)
        best_models[model_name] = grid_search.best_estimator_
        model_accuracies[model_name] = val_accuracy
    
    # 自动选择验证集上表现最好的模型
    best_model_name = max(model_accuracies, key=model_accuracies.get)
    best_model = best_models[best_model_name]
    print(f"\n选择的最佳模型是: {best_model_name}，验证集准确率: {model_accuracies[best_model_name]:.4f}")
    
    # 对测试集进行预测
    print(f"\n========== 使用最佳模型 '{best_model_name}' 进行测试集预测 ==========")
    test_predictions = best_model.predict(X_test)
    
    # 保存预测结果
    output_path = './data/test_predictions_optimized.csv'
    save_predictions(test_predictions, output_path)
    
    # 另外，您也可以尝试集成方法（Voting Classifier）来进一步提升性能
    # 以下是一个简单的集成示例
    print("\n========== 尝试集成方法 (Voting Classifier) ==========")
    
    # 定义个体模型管道
    svm_pipeline = build_pipeline(selected_model='SVM')
    logreg_pipeline = build_pipeline(selected_model='LogisticRegression')
    rf_pipeline = build_pipeline(selected_model='RandomForest')
    
    # 定义参数网格（这里只是示例，实际可能需要更复杂的调优）
    svm_param_grid = {'classifier__C': [0.1, 1, 10]}
    logreg_param_grid = {'classifier__C': [0.1, 1, 10], 'classifier__penalty': ['l2']}
    rf_param_grid = {'classifier__n_estimators': [100, 200], 'classifier__max_depth': [None, 10, 20]}
    
    # 对每个个体模型进行GridSearch
    print("正在调优 SVM...")
    grid_svm = GridSearchCV(svm_pipeline, svm_param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_svm.fit(X_train, y_train)
    print(f"SVM最佳参数: {grid_svm.best_params_}")
    
    print("正在调优 Logistic Regression...")
    grid_logreg = GridSearchCV(logreg_pipeline, logreg_param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_logreg.fit(X_train, y_train)
    print(f"Logistic Regression最佳参数: {grid_logreg.best_params_}")
    
    print("正在调优 Random Forest...")
    grid_rf = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, n_jobs=-1, scoring='accuracy', verbose=1)
    grid_rf.fit(X_train, y_train)
    print(f"Random Forest最佳参数: {grid_rf.best_params_}")
    
    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[
            ('svm', grid_svm.best_estimator_),
            ('logreg', grid_logreg.best_estimator_),
            ('rf', grid_rf.best_estimator_)
        ],
        voting='soft'  # 软投票
    )
    
    # 训练投票分类器
    print("正在训练 Voting Classifier...")
    voting_clf.fit(X_train, y_train)
    print("Voting Classifier 训练完成。")
    
    # 在验证集上评估投票分类器
    # 拆分训练集和验证集
    X_train_part, X_val, y_train_part, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    val_predictions = voting_clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"投票分类器 验证集准确率: {val_accuracy:.4f}")
    print("分类报告:")
    print(classification_report(y_val, val_predictions))
    
    # 对测试集进行预测并保存结果
    print("正在对测试集进行预测 (Voting Classifier)...")
    test_predictions_voting = voting_clf.predict(X_test)
    
    # 保存投票分类器的预测结果
    output_path_voting = './data/model_voting.csv'
    save_predictions(test_predictions_voting, output_path_voting)
    
    print("\n========== 所有模型训练与预测完成 ==========")

if __name__ == "__main__":
    main()
