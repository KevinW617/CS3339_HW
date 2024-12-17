import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from sklearn.utils import resample

def bias_variance_analysis(model, X_train, y_train, n_iterations=100):
    """计算模型的偏差和方差"""
    predictions = []
    
    # 划分固定的测试集
    X_train_main, X_test, y_train_main, y_test = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Bootstrap采样
    for i in range(n_iterations):
        X_boot, y_boot = resample(X_train_main, y_train_main, random_state=i)
        model_clone = clone(model)
        model_clone.fit(X_boot, y_boot)
        y_pred = model_clone.predict(X_test)
        predictions.append(y_pred)
    
    predictions = np.array(predictions)
    y_pred_mean = np.mean(predictions, axis=0)
    
    bias = np.mean((y_test - y_pred_mean) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    total_error = bias + variance
    
    return {
        'bias': bias,
        'variance': variance,
        'total_error': total_error
    }

def cross_validation(model, X_train, y_train, k_folds=5):
    """执行k折交叉验证"""
    cv_scores = []
    kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for train_idx, val_idx in kf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        model_clone = clone(model)
        model_clone.fit(X_fold_train, y_fold_train)
        y_pred = model_clone.predict(X_fold_val)
        err = np.sum(y_pred != y_fold_val)
        cv_scores.append(err / len(y_fold_val))
    
    return {
        'cv_scores': cv_scores,
        'mean_cv_score': np.mean(cv_scores),
        'std_cv_score': np.std(cv_scores)
    }

def calculate_aic_bic(model, X_train, y_train):
    """计算AIC和BIC"""
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    # 计算参数数量
    if hasattr(model, 'coef_'):
        n_params = model.coef_.size
    else:
        # 对于MLP，估计参数数量
        n_params = n_features * n_classes
    
    # 计算对数似然
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_train)
        log_likelihood = np.sum(np.log(y_prob[np.arange(len(y_train)), y_train]))
    else:
        # 对于SVM，使用多类别的hinge loss近似
        decision_values = model.decision_function(X_train)
        if decision_values.ndim == 2:
            # 多类别情况
            y_true_one_hot = np.zeros_like(decision_values)
            y_true_one_hot[np.arange(len(y_train)), y_train] = 1
            margins = decision_values * y_true_one_hot
            log_likelihood = -np.sum(np.maximum(0, 1 - margins))
        else:
            # 二分类情况
            log_likelihood = -np.sum(np.maximum(0, 1 - y_train * decision_values))
    
    # 计算AIC和BIC
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(n_samples) * n_params - 2 * log_likelihood
    
    return {
        'aic': aic,
        'bic': bic,
        'n_params': n_params,
        'log_likelihood': log_likelihood
    }

def evaluate_model(model, X_train, y_train):
    """综合评估模型"""
    # 偏差-方差分析
    bv_metrics = bias_variance_analysis(model, X_train, y_train)
    
    # 交叉验证
    cv_metrics = cross_validation(model, X_train, y_train)
    
    # AIC/BIC
    ic_metrics = calculate_aic_bic(model, X_train, y_train)
    
    return {
        'bias_variance': bv_metrics,
        'cross_validation': cv_metrics,
        'information_criteria': ic_metrics
    } 