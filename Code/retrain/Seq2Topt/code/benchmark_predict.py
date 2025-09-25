import math
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from model import MultiAttModel
import os
import warnings
import esm
import copy
from scipy.stats import pearsonr  # 用于PCC计算

def get_rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def get_r2(y_true, y_pred):
    from sklearn.metrics import r2_score
    return r2_score(y_true, y_pred)

def get_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rescale_targets(targets, min_val, max_val):
    # 假设原 targets 是归一化的，这里还原成原始值
    return np.array(targets) * (max_val - min_val) + min_val

def re_rescale_targets(targets, min_val, max_val):
    return np.array(targets) / (max_val - min_val)

def create_data_pack(df, task, embedding_dict, rparams):
    return [
        np.array(df.index),
        [embedding_dict[seq] for seq in df['sequence']],
        np.array(list(df[task]))
    ]

def test(model, test_pack, device):
    model.eval()
    predictions, target_values = [], []

    with torch.no_grad():
        for i in range(len(test_pack[0])):
            batch_data = [test_pack[di][i:i+1] for di in range(3)]
            ids, emb_list, y = batch_data

            # 转换维度: (1, seq_len, 320) → (1, 320, seq_len)
            emb = torch.tensor(emb_list[0], dtype=torch.float32).unsqueeze(0).to(device)
            emb = emb.permute(0, 2, 1)

            pred = model(emb)
            predictions.append(pred.item())
            target_values.append(y[0])
    predictions = np.array(predictions)
    # predictions = rescale_targets(predictions, *rparams[task])
    target_values = np.array(target_values)
    print(target_values)
    target_values = re_rescale_targets(target_values, *rparams[task])
    print(predictions)
    print(target_values)
    rmse = get_rmse(target_values, predictions)
    r2 = get_r2(target_values, predictions)
    mae = get_mae(target_values, predictions)
    pcc = pearsonr(target_values, predictions)[0]

    return rmse, r2, mae, pcc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference using a pre-trained model')
    parser.add_argument('--task', choices=['topt','tm','pHopt'], required=True)
    parser.add_argument('--test_path', required=True)
    parser.add_argument('--seq_model', required=True, help='PLM model name')
    parser.add_argument('--model_weight_path', required=True, help='Path to trained model weights (.pth)')
    parser.add_argument('--cv', default=None, help='Cross-validation fold number')
    args = parser.parse_args()

    test_path = str(args.test_path)
    task = str(args.task)
    model_weight_path = str(args.model_weight_path)

    if args.cv is not None:
        test_path = test_path.replace('.csv', f'_cv{args.cv}.csv')
        model_weight_path = model_weight_path.replace('.pth', f'_cv{args.cv}.pth')

    print('The test path is ' + test_path)
    print('The task is ' + task + '!')

    test_data = pd.read_csv(test_path)

    rparams = {'topt': (0, 120), 'tm': (0, 100), 'pHopt': (0, 14)}

    # Load ESM embeddings
    if args.task == 'topt':
        with open(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/pretrain/topt/{args.seq_model}_L.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)
    elif args.task == 'tm':
        with open(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/pretrain/tm_new/{args.seq_model}_L.pkl', 'rb') as f:
            embedding_dict = pickle.load(f)

    test_pack = create_data_pack(test_data, task, embedding_dict, rparams)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    emb_dim_dict = {
        'esm1b': 1280, 'esm1v': 1280, 'esm2': 1280,
        'esmc': 1152, 'prott5': 1024, 'prollama': 4096, 'esm8M': 320
    }
    emb_dim = emb_dim_dict[args.seq_model]
    n_head = 4
    n_RD = 4

    win_size = 3

    model = MultiAttModel(emb_dim, win_size, n_head, n_RD).to(device)
    
    # 加载预训练权重
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    print(f'Model loaded from {model_weight_path}')

    # 测试并计算性能指标
    rmse, r2, mae, pcc = test(model, test_pack, device)

    print(f"Test Results for Task '{task}':")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"PCC: {pcc:.4f}")