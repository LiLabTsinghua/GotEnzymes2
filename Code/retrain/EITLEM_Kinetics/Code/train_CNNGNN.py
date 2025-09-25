from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer
from KCM_CNNGNN import EitlemKcatPredictor  # 修改为新的模型文件
from ensemble_CNNGNN import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataLoader
import os
import shutil
import argparse
import pandas as pd
import pickle
from torch_geometric.data import Batch, Dataset, Data
import torch.nn.functional as F
import math
from sklearn.model_selection import train_test_split
import resource
import time

import math
import json
import pickle
import numpy as np
import torch
from collections import defaultdict
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
word_dict = defaultdict(lambda: len(word_dict))
atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def split_sequence(sequence, ngram):
    sequence = '-' + sequence + '='
    # print(sequence)
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def create_atoms(mol):
    """Create a list of atom (e.g., hydrogen and oxygen) IDs
    considering the aromaticity."""
    # atom_dict = defaultdict(lambda: len(atom_dict))
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)

def create_ijbonddict(mol):
    """Create a dictionary, which each key is a node ID
    and each value is the tuples of its neighboring node
    and bond (e.g., single and double) IDs."""
    # bond_dict = defaultdict(lambda: len(bond_dict))
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict

def extract_fingerprints(atoms, i_jbond_dict, radius):
    """Extract the r-radius subgraphs (i.e., fingerprints)
    from a molecular graph using Weisfeiler-Lehman algorithm."""

    # fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    # edge_dict = defaultdict(lambda: len(edge_dict))

    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges
            (i.e., r-radius subgraphs or fingerprints)."""
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(fingerprint_dict[fingerprint])
            nodes = fingerprints

            """Also update each edge ID considering two nodes
            on its both sides."""
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))

                    edge = edge_dict[(both_side, edge)]
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict
    return np.array(fingerprints)

def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as file:
        pickle.dump(dict(dictionary), file)

def tensor_long(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.long) for sublist in array_data]
    return tensor_list
def tensor_float(array_data):
    tensor_list = [torch.tensor(sublist, dtype=torch.float) for sublist in array_data]
    return tensor_list

class EitlemDataSet(Dataset):
    def __init__(self, data, sequence_embedding, smiles_embedding, log10=False):
        super().__init__()
        self.data = data
        self.sequence_embedding = sequence_embedding
        self.smiles_embedding = smiles_embedding
        self.log10 = log10
        self.use_cnn = True if self.sequence_embedding is None else False
        self.use_gnn = True if self.smiles_embedding is None else False
        self.max_len = 1024
        self.rows = [row for _, row in data.iterrows()]
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        pro_id = row['Sequence']
        smi_id = row['Smiles']
        value = float(row['Value'])

        # Apply logarithmic transformation
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)

        if self.use_cnn:
            protein_emb = split_sequence(pro_id, 3)
            pro_emb_tensor = torch.as_tensor(protein_emb, dtype=torch.long).view(-1)
            pro_batch = torch.zeros(len(protein_emb), dtype=torch.long)
        else:
            protein_emb = self.sequence_embedding.get(pro_id)
            pro_emb_tensor = torch.FloatTensor(protein_emb)
            pro_batch = None
            
        
        if self.use_gnn:
            mol = Chem.AddHs(Chem.MolFromSmiles(smi_id))
            atoms = create_atoms(mol)
            i_jbond_dict = create_ijbonddict(mol)
            fingerprints = extract_fingerprints(atoms, i_jbond_dict, 2)
            adjacency = create_adjacency(mol)
            # smiles_emb = (fingerprints, adjacency)
            fingerprints = torch.as_tensor(fingerprints, dtype=torch.long)
            adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
            # print('adjacency', adjacency)
            # print('adjacency.shape', adjacency.shape)
            # print('fp', fingerprints.shape)
            # num_fingerprints = int(fingerprints.max()) + 1
            # x_tensor = F.one_hot(torch.tensor(fingerprints, dtype=torch.long), num_classes=num_fingerprints).float()
            x_tensor = fingerprints
            adjacency_tensor = adjacency
        else:
            smiles_emb = self.smiles_embedding.get(smi_id)
            x_tensor = torch.as_tensor(smiles_emb, dtype=torch.float32).unsqueeze(0)
            adjacency_tensor = None

        value_tensor = torch.tensor(value, dtype=torch.float32)
        # print('process', adjacency_tensor.nonzero().t().contiguous())
        # print('1 shape', x_tensor.shape)
        # print('2 shape', adjacency_tensor.nonzero().t().contiguous().shape)
        # Create a Data object
        data = Data(
            x=x_tensor,
            edge_index=adjacency_tensor.nonzero().t().contiguous() if adjacency_tensor is not None else None,
            pro_emb=pro_emb_tensor,
            value=value_tensor,
            use_cnn=self.use_cnn,
            use_gnn=self.use_gnn,
            pro_batch=pro_batch
        )
        return data

    # def collate_fn(self, batch):
    #     return Batch.from_data_list(batch, follow_batch=['pro_emb'])
    @staticmethod
    def collate_fn(batch):
        batch = Batch.from_data_list(batch)
        
        if not hasattr(batch, 'pro_emb_batch'):
            pro_emb_lengths = []
            for data in batch.to_data_list():
                # 严格验证序列维度
                # assert data.pro_emb.dim() == 1, f"pro_emb 必须是一维序列，实际维度: {data.pro_emb.shape}"
                pro_emb_lengths.append(data.pro_emb.size(0))
            
            # # 验证总长度一致性
            # total_pro_len = sum(pro_emb_lengths)
            # assert batch.pro_emb.size(0) == total_pro_len, \
            #     f"pro_emb 总长度不匹配: 实际 {batch.pro_emb.size(0)} vs 预期 {total_pro_len}"
            
            # 生成批次索引
            batch.pro_emb_batch = torch.cat([
                torch.full((length,), i, dtype=torch.long) 
                for i, length in enumerate(pro_emb_lengths)
            ]).to(batch.pro_emb.device)
        
        return batch
    def __len__(self):
        return len(self.data)

def get_pair_info(data, test_flag_col='Test'):
    train_data = data[data[test_flag_col] == 0]
    test_data = data[data[test_flag_col] == 1]
    return train_data, test_data

def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device, smi_model, seq_model, kinetic_parameter, smiles_embedding, sequence_embedding):
    
    dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 167, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512, 'gnn': 128}
    dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096, 'esmc': 1152, 'cnn': 256}
    smi_shape = dict1[smi_model]
    seq_shape = dict2[seq_model]
    csv_path = f'../../../data/EITLEM_{kinetic_parameter}.csv'
    data = pd.read_csv(csv_path)
    data = data[~data['Smiles'].str.contains(r'\.')]
    train_pair_info, test_pair_info = get_pair_info(data)

    train_set = EitlemDataSet(train_pair_info, sequence_embedding, smiles_embedding, log10)
    test_set = EitlemDataSet(test_pair_info, sequence_embedding, smiles_embedding, log10)

    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=False)

    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}"

    # if os.path.exists(f'../Results/{Type}/{train_info}'):
    #     return None
    
    if kkmPath is not None:
        Epoch = 40 // (Iteration // 2)
    else:
        Epoch = 100
    
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        if Type == 'KCAT':
            model = EitlemKcatPredictor(
                mol_in_dim=smi_shape,
                hidden_dim=512,
                protein_dim=seq_shape,
                layer=10,
                dropout=0.5,
                att_layer=10
            )
            weights = model.state_dict()
            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
        else:
            model = EitlemKcatPredictor(
                mol_in_dim=smi_shape,
                hidden_dim=512,
                protein_dim=seq_shape,
                layer=10,
                dropout=0.5,
                att_layer=10
            )
            weights = model.state_dict()
            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
    else:
        if Type == 'KCAT':
            model = EitlemKcatPredictor(
                mol_in_dim=smi_shape,
                hidden_dim=512,
                protein_dim=seq_shape,
                layer=10,
                dropout=0.5,
                att_layer=10
            )
        else:
            model = EitlemKcatPredictor(
                mol_in_dim=smi_shape,
                hidden_dim=512,
                protein_dim=seq_shape,
                layer=10,
                dropout=0.5,
                att_layer=10
            )

    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'

    model = model.to(device)
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x:id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
            {'params': rest_param, 'lr':1e-4},
            {'params':model.out.parameters(), 'lr':1e-3},
        ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    
    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train, pcc_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration} iter epoch {epoch} train")
        if epoch % 5 == 0:
            MAE_dev, RMSE_dev, R2_dev, loss_dev, pcc_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration} iter epoch {epoch} valid")
        scheduler.step()
        if epoch % 5 == 0:
            writer.add_scalars("loss", {'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
            writer.add_scalars("RMSE", {'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
            writer.add_scalars("MAE", {'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
            writer.add_scalars("R2", {'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
            writer.add_scalars("PCC", {'train_PCC':pcc_train, 'dev_PCC':pcc_dev}, epoch)
            tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}_PCC_{pcc_dev:.4f}')
    pass

def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device, smi_model, seq_model, smiles_embedding, sequence_embedding):
    kinetic_parameter = 'KKM'
    dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 167, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512, 'gnn': 128}
    dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096, 'esmc': 1152, 'cnn': 256}
    smi_shape = dict1[smi_model]
    seq_shape = dict2[seq_model]
    csv_path = f'../../../data/EITLEM_{kinetic_parameter}.csv'
    data = pd.read_csv(csv_path)
    data = data[~data['Smiles'].str.contains(r'\.')]
    train_pair_info, test_pair_info = get_pair_info(data)

    train_set = EitlemDataSet(train_pair_info, sequence_embedding, smiles_embedding, log10)
    test_set = EitlemDataSet(test_pair_info, sequence_embedding, smiles_embedding, log10)

    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    train_info = f"Transfer-{TrainType}-KKM-train-{Iteration}-{smi_model}-{seq_model}"
    # if os.path.exists(f'../Results/KKM/{train_info}'):
    #     return None
    
    Epoch = 40
    file_model = f'../Results/KKM/{train_info}/Weight/'
    use_cnn = True if sequence_embedding is None else False
    use_gnn = True if smiles_embedding is None else False
    model = ensemble(smi_shape, 512, seq_shape, 10, 0.5, 10, use_cnn, use_gnn)
    kcat_pretrained = torch.load(kcatPath, weights_only=True)
    km_pretrained = torch.load(kmPath, weights_only=True)
    kcat_parameters = model.kcat.state_dict()
    km_parameters = model.km.state_dict()
    pretrained_kcat_para = {k:v for k, v in kcat_pretrained.items() if k in kcat_parameters}
    pretrained_km_para = {k:v for k, v in km_pretrained.items() if k in km_parameters}
    kcat_parameters.update(pretrained_kcat_para)
    km_parameters.update(pretrained_km_para)
    model.kcat.load_state_dict(kcat_parameters)
    model.km.load_state_dict(km_parameters)
    if not os.path.exists(file_model):
        os.makedirs(file_model)

    file_model += 'Eitlem_'
    model = model.to(device)
    optimizer = torch.optim.AdamW([
        {'params': model.kcat.parameters(), 'lr':1e-4},
        {'params':model.km.parameters(), 'lr':1e-4},
        {'params':model.o.parameters(), 'lr':1e-3},
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    print("start to training...")
    writer = SummaryWriter(f'../Results/KKM/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        train_MAE, train_rmse, train_r2, loss_train, pcc_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration} iter epoch {epoch} train")
        if epoch % 5 == 0:
            MAE_dev, RMSE_dev, R2_dev, loss_dev, pcc_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration} iter epoch {epoch} valid")
        scheduler.step()
        if epoch % 5 == 0:
            writer.add_scalars("loss", {'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
            writer.add_scalars("RMSE", {'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
            writer.add_scalars("MAE", {'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
            writer.add_scalars("R2", {'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
            writer.add_scalars("PCC", {'train_PCC':pcc_train, 'dev_PCC':pcc_dev}, epoch)
            tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}_PCC_{pcc_dev:.4f}')
    pass

def getPath(Type, TrainType, Iteration, smi_model, seq_model):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])

def TransferLearing(Iterations, smi_model, seq_model, TrainType, log10=False, molType='MACCSKeys', device=None):
    if os.path.exists(f'../Results/KKM/Transfer-{TrainType}-KKM-train-1-{smi_model}-{seq_model}'):
        sys.exit(f"{smi_model}-{seq_model} already exists.")
    else:
        print(f"smi_model: {smi_model}, seq_model: {seq_model}")
    # print('loading start')
    start_time = time.time()
    if seq_model != 'cnn':
        sequence_embedding = load_input_from_pkl(f'../../../pretrain/saved_models/{seq_model}_L.pkl')
    else:
        sequence_embedding = None
    if smi_model != 'gnn':
        smiles_embedding = load_input_from_pkl(f'../../../pretrain/saved_models/{smi_model}.pkl')
    else:
        smiles_embedding = None
    end_time = time.time()
    time_spent = end_time - start_time
    print(f'loading completed, total time = {time_spent}')
    for iteration in range(1, Iterations + 1):
        if iteration == 1:
            print('Now KCAT Training')
            kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device, smi_model, seq_model, 'KCAT', smiles_embedding, sequence_embedding)
            print('Now KM Training')
            kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device, smi_model, seq_model, 'KM', smiles_embedding, sequence_embedding)
        else:
            kkmPath = getPath('KKM', TrainType, iteration-1)
            kineticsTrainer(kkmPath, TrainType, 'KCAT', iteration, log10, molType, device, smi_model, seq_model, 'KCAT')
            kineticsTrainer(kkmPath, TrainType, 'KM', iteration, log10, molType, device, smi_model, seq_model, 'KM')
        
        kcatPath = getPath(f'KCAT', TrainType, iteration, smi_model, seq_model)
        kmPath = getPath(f'KM', TrainType, iteration, smi_model, seq_model)
        print('Now KKM Training')
        KKMTrainer(kcatPath, kmPath, TrainType, iteration, log10, molType, device, smi_model, seq_model, smiles_embedding, sequence_embedding)
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--Iteration', type=int, required=True)
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, required=False, default=True)
    parser.add_argument('-m', '--molType', type=str, required=False, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    parser.add_argument('-smi', '--smi_model', type=str, required=True)
    parser.add_argument('-seq', '--seq_model', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    new_limit = 400 * 1024 * 1024 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (new_limit, -1))
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    # print(f"used device {device}")
    TransferLearing(args.Iteration, args.smi_model, args.seq_model, args.TrainType, args.log10, args.molType, device)
