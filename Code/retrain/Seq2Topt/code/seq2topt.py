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
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr

'''
Predict enzyme optimal temperature.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict enzyme Topt from protein sequences. \
                                    Inputs: --input: the path of input dataset(csv); \
                                    --output: output path of prediction result.')
    
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    parser.add_argument('--cv', required = True)
    parser.add_argument('--type0', required = True, choices=['topt', 'tm'])
    args = parser.parse_args()
    
    # topt_pth = '../model_topt_window.3_r2.0.57.pth';
    emb_model = 'prott5'
    # args.output = f'../data/' + args.input.replace('.csv', f'_{emb_model}_{args.type0}_cv{args.cv}')
    topt_pth = f'../data/zuhe/{emb_model}_{args.type0}_new_best_cv{args.cv}.pth' # 注意修改为不是window的版本
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')
    emb_dim_dict = {'esm1b': 1280, 'esm1v': 1280, 'esm2': 1280, 'esmc': 1152, 'prott5': 1024, 'prollama': 4096, 'esm8M': 320}
    emb_dim= emb_dim_dict[emb_model]
    window=3; n_head = 4; n_RD = 4;
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    model = MultiAttModel( emb_dim, window, n_head, n_RD)
    model.to(device);
    model.load_state_dict(torch.load(topt_pth, map_location=device, weights_only=True))
    model.eval()
    
    input_data = pd.read_csv(str(args.input))
    # batch_size=4
    # #Load esm2
    # esm2_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D() # 6 layers
    # esm2_model = esm2_model.to(device)
    # esm2_batch_converter = alphabet.get_batch_converter()
    predictions = []
    # for i in range( math.ceil( len(input_data.index) / batch_size ) ):
    #     ids = list(input_data.index)[i * batch_size: (i + 1) * batch_size]
    #     seqs = list(input_data['sequence'])[i * batch_size: (i + 1) * batch_size]
    #     #embeddings
    #     inputs = [(ids[i], seqs[i]) for i in range(len(ids))]
    #     batch_labels, batch_strs, batch_tokens = esm2_batch_converter(inputs)
    #     batch_tokens = batch_tokens.to(device=device, non_blocking=True)
    #     with torch.no_grad():
    #         emb = esm2_model(batch_tokens, repr_layers=[6], return_contacts=False)
    #     emb = emb["representations"][6]
    #     emb = emb.transpose(1,2)
    #     emb = emb.to(device)
    sequence = list(input_data['sequence'])
    emb_dict = pickle.load(open(f'/home/wuke/project/bio_deeplearning/zzz_benchmark/pretrain/{args.type0}_new/{emb_model}_L.pkl', 'rb'))# _new
    for seq in sequence:
        emb_tensor = torch.from_numpy(emb_dict[seq])
        emb_tensor = emb_tensor.transpose(0,1).unsqueeze(0)
        emb_tensor = emb_tensor.to(device)
        with torch.no_grad():
            preds = model(emb_tensor)
        predictions += preds.cpu().detach().numpy().reshape(-1).tolist()
    
    Topt_max = 100 if args.type0 == 'tm' else 120  
    # print(Topt_max)
    pred_topts = [float(v*Topt_max) for v in predictions ]
    result_pd = pd.DataFrame(zip(list(input_data.index), list(input_data['sequence']), list(input_data[f'{args.type0}']), [Topt_max - pred for pred in pred_topts]), columns=['id','sequence',f'{args.type0}', f'pred_{args.type0}'])
    
    result_pd.to_csv( str( args.output ) +'.csv' ,index=None)
    # Normalize topt and predicted values
    true_values = result_pd[f'{args.type0}'] / Topt_max
    predicted_values = result_pd[f'pred_{args.type0}'] / Topt_max
    # print(true_values, predicted_values)
    # Calculate metrics
    pcc, _ = pearsonr(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    rmse = root_mean_squared_error(true_values, predicted_values)
    mae = mean_absolute_error(true_values, predicted_values)

    # Print metrics
    print(f'{args.cv} PCC: {pcc:.4f} R2: {r2:.4f} RMSE: {rmse:.4f} MAE: {mae:.4f}')
    # print('Task '+ str(args.input)+' completed!')
    
    
    
    
    