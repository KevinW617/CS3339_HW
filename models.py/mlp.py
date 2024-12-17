import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import issparse, csr_matrix
import warnings
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from models.model_evaluation import evaluate_model
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

class TextDataset(Dataset):
    """自定义数据集"""
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features.toarray())  # 转换稀疏矩阵为密集张量
        self.labels = torch.LongTensor(labels) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

class SimpleMLPClassifier(nn.Module):
    """简单的MLP分类器，只使用一个隐藏层"""
    def __init__(self, input_dim, hidden_dim=256, num_classes=20):
        super(SimpleMLPClassifier, self).__init__()
        self.network = nn.Sequential(
            # 输入层 -> 隐藏层
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.9),
            # 隐藏层 -> 输出层
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

class MLPWrapper(BaseEstimator, ClassifierMixin):
    """包装MLP模型以兼容scikit-learn的API"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def fit(self, X, y):
        return self
        
    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X.toarray()).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probas = torch.softmax(outputs, dim=1)
        return probas.cpu().numpy()

def load_data(train_feature_path, train_label_path, test_feature_path):
    """加载数据"""
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

def train_model(X_train, y_train, device):
    """训练MLP模型并进行评估"""
    # 划分训练集和验证集
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 创建数据加载器
    train_dataset = TextDataset(X_train_split, y_train_split)
    val_dataset = TextDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 初始化模型
    model = SimpleMLPClassifier(input_dim=X_train.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("训练MLP模型...")
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
                outputs = model(batch_features)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Loss: {total_loss/len(train_loader):.4f}, '
                  f'Train Acc: {train_correct}/{train_total} = {train_correct/train_total:.4f}, '
                  f'Val Acc: {val_correct}/{val_total} = {val_correct/val_total:.4f}')
    
    # 最终评估
    model.eval()
    with torch.no_grad():
        # 训练集性能
        train_preds = []
        train_true = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            train_preds.extend(predicted.cpu().numpy())
            train_true.extend(batch_labels.numpy())
        
        train_acc = accuracy_score(train_true, train_preds)
        print(f"\n最终训练集准确率: {sum(np.array(train_preds) == np.array(train_true))}/{len(train_true)} = {train_acc:.4f}")
        
        # 验证集性能
        val_preds = []
        val_true = []
        
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_labels.numpy())
        
        val_acc = accuracy_score(val_true, val_preds)
        print(f"\n最终验证集准确率: {sum(np.array(val_preds) == np.array(val_true))}/{len(val_true)} = {val_acc:.4f}")
        print("\n验证集分类报告:")
        print(classification_report(val_true, val_preds))
    
    # 在训练完成后添加评估
    print("\n执行综合模型评估...")
    wrapped_model = MLPWrapper(model, device)
    evaluation_metrics = evaluate_model(wrapped_model, X_train, y_train)
    
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
    """保存预测结果"""
    submission_df = pd.DataFrame({
        'ID': range(len(predictions)),
        'label': predictions
    })
    submission_df.to_csv(output_path, index=False)
    print(f"预测结果已保存至 {output_path}")

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    data_dir = '../data'
    train_feature_path = os.path.join(data_dir, 'train_feature.pkl')
    train_label_path = os.path.join(data_dir, 'train_labels.npy')
    test_feature_path = os.path.join(data_dir, 'test_feature.pkl')
    
    # 检查文件
    for path in [train_feature_path, train_label_path, test_feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"未找到文件: {path}")
    
    # 加载数据
    X_train, y_train, X_test = load_data(train_feature_path, train_label_path, test_feature_path)
    
    # 训练模型
    model, evaluation_metrics = train_model(X_train, y_train, device)
    
    # 预测测试集
    print("\n预测测试集...")
    model.eval()
    test_dataset = TextDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_preds = []
    
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
    
    # 保存预测结果
    output_path = os.path.join(data_dir, 'test_predictions_mlp.csv')
    save_predictions(test_preds, output_path)
    
    print("\n========== 模型训练与预测完成 ==========")

if __name__ == "__main__":
    main() 