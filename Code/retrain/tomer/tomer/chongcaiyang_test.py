import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from resreg import bin_split, matthews_corrcoef, f1_score, bin_performance
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import resreg as resreg
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import resreg as resreg

# Variables
#=================#

aalist = list('ACDEFGHIKLMNPQRSTVWY')
this_dir, this_filename = os.path.split(__file__)

# Get TOMER model
#======================#

def getAAC(seq):
    '''Calculate amino acid composition for a protein sequences (string)'''
    aac = np.array([seq.count(x) for x in aalist])/len(seq)
    return aac
# 定义分箱边界（假设论文中的分箱为：[0-30, 30-50, 50-65, 65-85, 85-120])
bins = [30, 50, 65, 85]  # 分箱边界

# 加载数据
data = pd.read_csv('data/sequence_ogt_topt.csv')
X = np.array([getAAC(seq) for seq in data['sequence']])
y = data['topt'].values

# 分箱并获取每个bin的索引
bin_indices, bin_freqs = bin_split(y, bins)

def uniform_test_split(bin_indices, test_size=70, random_state=None):
    test_indices = []
    for indices in bin_indices:
        # 如果某个bin的样本数不足，跳过或调整test_size
        if len(indices) >= test_size:
            selected = np.random.choice(indices, test_size, replace=False)
            test_indices.extend(selected)
    train_indices = list(set(range(len(y))) - set(test_indices))
    # print(f"Train size: {len(train_indices)}, Test size: {len(test_indices)}")
    return train_indices, test_indices

n_iterations = 50
metrics = {'R2': [], 'MCC': [], 'F1': [], 'Bin_MSE': []}

for i in range(n_iterations):
    # 划分训练集和测试集
    train_idx, test_idx = uniform_test_split(bin_indices, test_size=70, random_state=i)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 标准化特征（仅用训练集）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 训练TOMER模型（使用Rebagg）
    relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=72.2)
    tomer_rebagg = resreg.Rebagg(m=100, s=600, base_reg=DecisionTreeRegressor())
    tomer_rebagg.fit(X_train, y_train, relevance=relevance, relevance_threshold=0.5,
                     sample_method='random_oversample', size_method='variation',
                     random_state=i)
    
    # 预测测试集
    y_pred = tomer_rebagg.predict(X_test)
    
    # 计算指标
    r2 = r2_score(y_test, y_pred)
    mcc = resreg.matthews_corrcoef(y_test, y_pred, bins=bins)
    relevance_test = resreg.sigmoid_relevance(y_test, cl=None, ch=72.2)
    f1 = resreg.f1_score(y_test, y_pred, error_threshold=5.0, 
                        relevance_true=relevance_test, 
                        relevance_pred=relevance_test, 
                        relevance_threshold=0.5)
    bin_mse = resreg.bin_performance(y_test, y_pred, bins=bins, metric='MSE')
    
    # 存储结果
    metrics['R2'].append(r2)
    metrics['MCC'].append(mcc)
    metrics['F1'].append(f1)
    metrics['Bin_MSE'].append(bin_mse)

# 计算平均性能
avg_r2 = np.mean(metrics['R2'])
std_r2 = np.std(metrics['R2'])
print(f"R²: {avg_r2:.3f} ± {std_r2:.3f}")