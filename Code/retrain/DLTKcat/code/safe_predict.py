import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from scipy import stats
import pickle
import argparse
import math
from math import sqrt
import numpy as np
import pandas as pd
from feature_functions import load_pickle
from train_functions import batch2tensor, load_data, scores
import os
import warnings
from DLTKcat import DLTKcat

def safe_load_data(data_path, has_label):
    try:
        return load_data(data_path, has_label)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def safe_batch2tensor(batch_data, has_label, device):
    try:
        return batch2tensor(batch_data, has_label, device)
    except Exception as e:
        print(f"Error in batch2tensor: {e}")
        return None

def is_valid_smiles(smiles):
    # Implement a simple SMILES validation check here
    # This function can be customized to validate SMILES strings as needed
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv); \
                                    --output: output path of prediction result; \
                                    --has_label: whether the input dataset(csv) has labels')

    parser.add_argument('--model_path', required=True)
    parser.add_argument('--param_dict_pkl', default='../data/hyparams/default.pkl')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--has_label', type=str, choices=['False', 'True'], default='True')
    args = parser.parse_args()
    
    has_label = args.has_label == 'True'
    param_dict = load_pickle(str(args.param_dict_pkl))
    atom_dict = load_pickle('../data/dict/fingerprint_dict.pkl')
    word_dict = load_pickle('../data/dict/word_dict.pkl')
    
    device = torch.device('cpu')
    print('Task ' + str(args.input) + ' started!')

    comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
                      param_dict['comp_dim'], param_dict['prot_dim'], param_dict['gat_dim'], param_dict['num_head'], \
                      param_dict['dropout'], param_dict['alpha'], param_dict['window'], param_dict['layer_cnn'], \
                      param_dict['latent_dim'], param_dict['layer_out']

    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    
    # Load model
    M = DLTKcat(len(atom_dict), len(word_dict), comp_dim, prot_dim, gat_dim, num_head, \
                dropout, alpha, window, layer_cnn, latent_dim, layer_out)
    M.to(device)
    M.load_state_dict(torch.load(str(args.model_path), map_location=device))
    
    # Prepare input
    if os.path.isdir('../data/pred/temp'):
        os.system('rm -rf ../data/pred/temp')
    os.system('mkdir ../data/pred/temp')
    
    os.system(f'python gen_features.py --data {str(args.input)} --output ../data/pred/temp/ --has_dict True --has_label {str(args.has_label)}')
    
    data_input = safe_load_data('../data/pred/temp/', has_label)
    if data_input is None:
        print('Failed to load input data.')
        sys.exit(1)
    
    predictions, labels, valid_indices = [], [], []
    batch_size = 16
    for i in range(math.ceil(len(data_input[0]) / batch_size)):
        batch_data = [data_input[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_input))]
        batch_tensor_data = safe_batch2tensor(batch_data, has_label, device)
        if batch_tensor_data is None:
            continue
        if has_label:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, inv_Temp, Temp, label = batch_tensor_data
        else:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad, amino_mask, inv_Temp, Temp = batch_tensor_data
            
        try:
            with torch.no_grad():
                pred = M(atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp)
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            valid_indices += list(range(i * batch_size, min((i + 1) * batch_size, len(data_input[0]))))
            if has_label:
                labels += label.cpu().numpy().reshape(-1).tolist()
        except Exception as e:
            print(f"Error during prediction: {e}")
            continue
    
    predictions = np.array(predictions)
    if has_label:
        labels = np.array(labels)
        rmse, r2 = scores(labels, predictions)
        print('Accuracy: RMSE=' + str(rmse) + ', R2=' + str(r2))
    else:
        print('No labels provided.')

    # Save prediction results
    data = pd.read_csv(str(args.input))
    data['pred_log10kcat'] = np.nan
    data.loc[valid_indices, 'pred_log10kcat'] = predictions
    data.to_csv(str(args.output) + '.csv', index=None)
    
    # Delete intermediate files
    os.system('rm -rf ../data/pred/temp')
    print('Task ' + str(args.input) + ' completed!')
