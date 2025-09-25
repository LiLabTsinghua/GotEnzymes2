import math
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from functions import *
from model import MultiAttModel
import os
import warnings
import random
import esm
import copy  # 新增copy模块用于深拷贝模型参数

'''
Run the training process. ESM-2
'''
def set_random_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_eval(model, train_pack, test_pack, dev_pack, device, lr, batch_size, lr_decay, decay_interval, num_epochs, patience=10, delta=0.0001):
    criterion = F.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=decay_interval, gamma=lr_decay)
    idx = np.arange(len(train_pack[0]))
    
    min_size = 4
    div_min = batch_size // min_size if batch_size > min_size else 1
    
    train_result = {'rmse_train':[], 'r2_train':[], 'mae_train':[],
                   'rmse_test':[], 'r2_test':[], 'mae_test':[],
                   'rmse_dev':[], 'r2_dev':[], 'mae_dev':[]}
    
    best_r2_dev = -np.inf  # 初始化最佳验证集R²
    epochs_no_improve = 0  # 未改善的epoch计数
    best_model_weights = None  # 保存最佳模型参数
    
    for epoch in range(num_epochs):
        np.random.shuffle(idx)
        model.train()
        predictions, targets = [], []
        
        for i in range(len(train_pack[0])):
            batch_data = [train_pack[di][i:i+1] for di in range(3)]  # 取单个样本
            ids, emb_list, y = batch_data
            
            # 转换维度 (1, seq_len, 320) → (1, 320, seq_len)
            emb = torch.tensor(emb_list[0], dtype=torch.float32).unsqueeze(0).to(device)
            # print(emb.shape)
            if emb.dim() == 3:
                emb = emb.permute(0, 2, 1)
            else:
                print(emb.shape)
            
            target_values = torch.FloatTensor(y).unsqueeze(1).to(device)
            
            pred = model(emb)
            loss = criterion(pred.float(), target_values.float())
            predictions.append(pred.item())
            targets.append(y[0])
            
            # 立即更新参数（batch_size=1时每次迭代都更新）
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        predictions = np.array(predictions); targets = np.array(targets);
        train_result['rmse_train'].append( get_rmse( targets, predictions) )
        train_result['r2_train'].append( get_r2( targets, predictions) )
        train_result['mae_train'].append( get_mae( targets, predictions) )
        
        rmse_test, r2_test, mae_test = test(model, test_pack, batch_size, device )
        rmse_dev, r2_dev, mae_dev = test(model, dev_pack, batch_size, device )
        train_result['rmse_test'].append(rmse_test); train_result['r2_test'].append(r2_test); train_result['mae_test'].append(mae_test);
        train_result['rmse_dev'].append(rmse_dev); train_result['r2_dev'].append(r2_dev); train_result['mae_dev'].append(mae_dev);
        
        # 早停机制检查
        current_r2_dev = r2_dev
        if current_r2_dev > best_r2_dev + delta:
            best_r2_dev = current_r2_dev
            epochs_no_improve = 0
            best_model_weights = copy.deepcopy(model.state_dict())  # 保存最佳模型参数
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered at epoch {epoch}, best R2_dev: {best_r2_dev:.4f}')
                model.load_state_dict(best_model_weights)  # 恢复最佳模型参数
                break  # 终止训练循环
        
        if epoch%5 == 0:
            print('epoch: '+str(epoch)+'/'+ str(num_epochs) +';  rmse test: ' + str(rmse_test) + '; r2 test: ' + str(r2_test) )
        
        scheduler.step()
    
    # 训练结束后确保加载最佳模型参数
    if best_model_weights is not None:
        model.load_state_dict(best_model_weights)
    else:
        best_model_weights = copy.deepcopy(model.state_dict())
    
    return train_result
            
def test(model, test_pack, batch_size, device):
    model.eval()
    predictions, target_values = [], []
    
    for i in range(len(test_pack[0])):
        batch_data = [test_pack[di][i:i+1] for di in range(3)]
        ids, emb_list, y = batch_data
        
        # 转换维度
        emb = torch.tensor(emb_list[0], dtype=torch.float32).unsqueeze(0).to(device)
        emb = emb.permute(0, 2, 1)
        
        with torch.no_grad():
            preds = model(emb)
        predictions.append(preds.item())
        target_values.append(y[0])
            
    predictions = np.array(predictions)
    target_values = np.array(target_values)
    rmse = get_rmse( target_values, predictions)
    r2 = get_r2( target_values, predictions)
    mae = get_mae( target_values, predictions)
    return rmse, r2, mae

def split_data(data, ratio=0.1):
    n_samples = len(data[0])
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    num_split = int(n_samples * ratio)
    idx_1, idx_0 = idx[:num_split], idx[num_split:]
    data_0 = [
        data[0][idx_0],
        [data[1][i] for i in idx_0],
        data[2][idx_0]
    ]
    data_1 = [
        data[0][idx_1],
        [data[1][i] for i in idx_1],
        data[2][idx_1]
    ]
    return data_0, data_1

def create_data_pack(df, task, embedding_dict):
    return [
        np.array(df.index),
        [embedding_dict[seq] for seq in df['sequence']],
        np.array(rescale_targets(list(df[task]), *rparams[task]))
    ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--task',choices=['topt','tm','pHopt'], required = True)
    parser.add_argument('--train_path', required = True)
    parser.add_argument('--test_path', required = True)
    parser.add_argument('--lr', default = 0.0005, type=float )
    parser.add_argument('--batch', default = 32 , type=int )
    parser.add_argument('--lr_decay', default = 0.5, type=float )
    parser.add_argument('--decay_interval', default = 10, type=int )
    parser.add_argument('--num_epoch', default = 30, type=int )
    parser.add_argument('--seq_model', required=True, help='PLM model name')
    # 添加早停参数
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait before early stopping')
    parser.add_argument('--delta', type=float, default=0.0001, help='Minimum improvement required to reset patience')
    parser.add_argument('--cv', default = None, help='cv number')
    args = parser.parse_args()
    
    set_random_seeds(0);
    
    train_path, test_path, lr, batch_size, lr_decay, decay_interval = \
            str(args.train_path), str(args.test_path), float(args.lr), int(args.batch), \
            float(args.lr_decay), int(args.decay_interval)
    
    task = str(args.task)
    if args.cv is not None:
        train_path = train_path.replace('.csv', f'_cv{args.cv}.csv')
        test_path = test_path.replace('.csv', f'_cv{args.cv}.csv')
    # print('The train path is '+ train_path)
    # print('The test path is '+ test_path)
    # print('The task is '+ task+'!')
   
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    rparams = {'topt':(0,120),'tm':(0,100),'pHopt':(0,14)}
    if args.task == 'topt':
        with open(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/pretrain/topt/{args.seq_model}_L.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
    elif args.task == 'tm':
        with open(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/pretrain/tm_new/{args.seq_model}_L.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
    train_pack = create_data_pack(train_data, task, embedding_dict)
    test_pack = create_data_pack(test_data, task, embedding_dict)
    train_pack, dev_pack = split_data( train_pack, 0.1)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(f'Using device: {device}')
    
    num_epochs = int(args.num_epoch)
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    emb_dim_dict = {'esm1b': 1280, 'esm1v': 1280, 'esm2': 1280, 'esmc': 1152, 'prott5': 1024, 'prollama': 4096, 'esm8M': 320}
    emb_dim= emb_dim_dict[args.seq_model]
    n_head = 4
    n_RD = 4

    win_size = 3
    M = MultiAttModel( emb_dim, win_size, n_head, n_RD)
    M.to(device)

    train_result = train_eval(
        M, train_pack, test_pack, dev_pack, device, 
        lr, batch_size, lr_decay, decay_interval, num_epochs,
        patience=args.patience, delta=args.delta  # 传递早停参数
    )
    train_result['Epoch'] = list(np.arange(1, len(train_result['rmse_train'])+1))
    output_path = os.path.join(f'../data/zuhe/{args.seq_model}_{task}_window{win_size}.csv')
    best_model_pth = f'../data/zuhe/{args.seq_model}_{task}_new_best.pth'
    if args.cv is not None:
        output_path = output_path.replace('.csv', f'_cv{args.cv}.csv')
        best_model_pth = best_model_pth.replace('.pth', f'_cv{args.cv}.pth')
    pd.DataFrame(train_result).to_csv(output_path, index=None)
    # 保存最佳模型
    torch.save(M.state_dict(), best_model_pth)
    print(f'Best model saved to {best_model_pth}')
    
    print('Done.')