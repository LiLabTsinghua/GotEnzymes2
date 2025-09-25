from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import EitlemDataLoader
import os
import shutil
import argparse
import pandas as pd
import pickle
from torch_geometric.data import Batch, Dataset, Data
import math
from sklearn.model_selection import train_test_split
import resource
import time

class EitlemDataSet(Dataset):
    def __init__(self, data, sequence_embedding, smiles_embedding, log10=False):
        super(EitlemDataSet, self).__init__()
        self.data = data
        self.sequence_embedding = sequence_embedding
        self.smiles_embedding = smiles_embedding
        self.log10 = log10
        # print(f"log10:{self.log10} molType:{self.Type}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pro_id = row['Sequence']
        smi_id = row['Smiles']
        value = row['Value']

        protein_emb = self.sequence_embedding[pro_id]
        smiles_emb = self.smiles_embedding[smi_id]

        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)
        data = Data(x = torch.FloatTensor(smiles_emb).unsqueeze(0), pro_emb=torch.FloatTensor(protein_emb), value=value)
        # print(data.x.shape, data.pro_emb.shape, data.value)
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])

    def __len__(self):
        return len(self.data)

def get_pair_info(data, test_flag_col='Test'):
    train_data = data[data[test_flag_col] == 0]
    test_data = data[data[test_flag_col] == 1]
    return train_data, test_data

# def get_pair_info(data):
#     train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
#     return train_data, test_data
def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device, smi_model, seq_model, kinetic_parameter, smiles_embedding, sequence_embedding):
    
    dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 167, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512}
    dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096, 'esmc': 1152}
    smi_shape = dict1[smi_model]
    seq_shape = dict2[seq_model]
    csv_path = f'../../../data/EITLEM_{kinetic_parameter}.csv'
    data = pd.read_csv(csv_path)
    train_pair_info, test_pair_info = get_pair_info(data)

    train_set = EitlemDataSet(train_pair_info, sequence_embedding, smiles_embedding, log10)
    test_set = EitlemDataSet(test_pair_info, sequence_embedding, smiles_embedding, log10)

    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=False)

    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}"

    if os.path.exists(f'../Results/{Type}/{train_info}'):
        return None
    
    if kkmPath is not None:
        Epoch = 40 // (Iteration // 2)
    else:
        Epoch = 100
    
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    
    if kkmPath is not None:
        trained_weights = torch.load(kkmPath)
        if Type == 'KCAT':
            model = EitlemKcatPredictor(smi_shape, 512, seq_shape, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[5:]: v for k, v in trained_weights.items() if 'kcat' in k and k[5:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
        else:
            model = EitlemKmPredictor(smi_shape, 512, seq_shape, 10, 0.5, 10)
            weights = model.state_dict()
            pretrained_para = {k[3:]: v for k, v in trained_weights.items() if 'km' in k and k[3:] in weights}
            weights.update(pretrained_para)
            model.load_state_dict(weights)
    else:
        if Type == 'KCAT':
            model = EitlemKcatPredictor(smi_shape, 512, seq_shape, 10, 0.5, 10)
        else:
            model = EitlemKmPredictor(smi_shape, 512, seq_shape, 10, 0.5, 10)

    if not os.path.exists(file_model):
        os.makedirs(file_model)
    file_model += 'Eitlem_'
    """Train setting."""
    # train_pair_info, test_pair_info = get_pair_info("../Data/", Type, False)
    # train_set= EitlemDataSet(train_pair_info, f'../Data/Feature/esm2_t33_650M_UR50D/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    # test_set = EitlemDataSet( test_pair_info, f'../Data/Feature/esm2_t33_650M_UR50D/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    # train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    # valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=5, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    if kkmPath is not None:
        out_param = list(map(id, model.out.parameters()))
        rest_param = filter(lambda x:id(x) not in out_param, model.parameters())
        optimizer = torch.optim.AdamW([
                                       {'params': rest_param, 'lr':1e-4},
                                       {'params':model.out.parameters(), 'lr':1e-3},
                                      ])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
    
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    
    # print("start to training...")
    writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        # continue
        train_MAE, train_rmse, train_r2, loss_train, pcc_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration} iter epoch {epoch} train:")
        if epoch % 5 == 0:
            MAE_dev, RMSE_dev, R2_dev, loss_dev, pcc_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration} iter epoch {epoch} valid:")
        scheduler.step()
        if epoch % 5 == 0:
            writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
            writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
            writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
            writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
            writer.add_scalars("PCC",{'train_PCC':pcc_train, 'dev_PCC':pcc_dev}, epoch)
            tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}_PCC_{pcc_dev:.4f}') # 保存
    pass


def KKMTrainer(kcatPath, kmPath, TrainType, Iteration, log10, molType, device, smi_model, seq_model, smiles_embedding, sequence_embedding):
    kinetic_parameter = 'KKM'
    dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 167, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512}
    dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096, 'esmc': 1152}
    smi_shape = dict1[smi_model]
    seq_shape = dict2[seq_model]
    csv_path = f'../../../data/EITLEM_{kinetic_parameter}.csv'
    data = pd.read_csv(csv_path)
    train_pair_info, test_pair_info = get_pair_info(data)

    train_set = EitlemDataSet(train_pair_info, sequence_embedding, smiles_embedding, log10)
    test_set = EitlemDataSet(test_pair_info, sequence_embedding, smiles_embedding, log10)

    train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=False)
    train_info = f"Transfer-{TrainType}-KKM-train-{Iteration}-{smi_model}-{seq_model}"
    if os.path.exists(f'../Results/KKM/{train_info}'):
        return None
    
    Epoch = 40
    file_model = f'../Results/KKM/{train_info}/Weight/'
    dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 167, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512}
    dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096}
    smi_shape = dict1[smi_model]
    seq_shape = dict2[seq_model]
    model = ensemble(smi_shape, 512, seq_shape, 10, 0.5, 10)
    kcat_pretrained = torch.load(kcatPath)
    km_pretrained = torch.load(kmPath)
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
    """Train setting."""
    # train_pair_info, test_pair_info = get_pair_info("../Data/", 'KKM')
    # train_set= EitlemDataSet(train_pair_info, f'../Data/Feature/esm2_t33_650M_UR50D/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    # test_set = EitlemDataSet( test_pair_info, f'../Data/Feature/esm2_t33_650M_UR50D/', f'../Data/Feature/index_smiles', 1024, 4, log10, molType)
    # train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    # valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=60, prefetch_factor=10, persistent_workers=True, pin_memory=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW([
                                   {'params': model.kcat.parameters(), 'lr':1e-4},
                                   {'params':model.km.parameters(), 'lr':1e-4},
                                   {'params':model.o.parameters(), 'lr':1e-3},
                                  ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.9)
    loss_fn = nn.MSELoss()
    tester = Tester(device, loss_fn, log10=log10)
    trainer = Trainer(device, loss_fn, log10=log10)
    print("start to training...")
    writer = SummaryWriter(f'../Results/KKM/{train_info}/logs/')
    for epoch in range(1, Epoch + 1):
        # continue
        train_MAE, train_rmse, train_r2, loss_train, pcc_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration} iter epoch {epoch} train:")
        if epoch % 5 == 0:
            MAE_dev, RMSE_dev, R2_dev, loss_dev, pcc_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration} iter epoch {epoch} valid:")
        scheduler.step()
        if epoch % 5 == 0:
            writer.add_scalars("loss",{'train_loss':loss_train, 'dev_loss':loss_dev}, epoch)
            writer.add_scalars("RMSE",{'train_RMSE':train_rmse, 'dev_RMSE':RMSE_dev}, epoch)
            writer.add_scalars("MAE",{'train_MAE':train_MAE, 'dev_MAE':MAE_dev}, epoch)
            writer.add_scalars("R2",{'train_R2':train_r2, 'dev_R2':R2_dev}, epoch)
            writer.add_scalars("PCC",{'train_PCC':pcc_train, 'dev_PCC':pcc_dev}, epoch)
            tester.save_model(model, file_model+f'{molType}_trainR2_{train_r2:.4f}_devR2_{R2_dev:.4f}_RMSE_{RMSE_dev:.4f}_MAE_{MAE_dev:.4f}_PCC_{pcc_dev:.4f}') # 保存
    pass


def getPath(Type, TrainType, Iteration, smi_model, seq_model):
    train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}"
    file_model = f'../Results/{Type}/{train_info}/Weight/'
    fileList = os.listdir(file_model)
    return os.path.join(file_model, fileList[0])

def TransferLearing(Iterations, smi_model, seq_model, TrainType, log10=False, molType='MACCSKeys', device=None):
    # smi_model_list = ['maccskeys', 'smitrans', 'molgen', 'molebert', 'unimolv1', 'unimolv2', 'chemberta2', 'ecfp', 'rdkitfp']
    # seq_model_list = ['esm2', 'esm1b', 'esm3b', 'esm15b', 'prott5', 'prollama']
    # for smi_model in smi_model_list:
    #     for seq_model in seq_model_list:
            if os.path.exists(f'../Results/KKM/Transfer-{TrainType}-KKM-train-1-{smi_model}-{seq_model}'):
                sys.exit(f"{smi_model}-{seq_model} already exists.")
            else:
                print(f"smi_model: {smi_model}, seq_model: {seq_model}")
            print('loading start')
            start_time = time.time()
            sequence_embedding = load_input_from_pkl(f'../../../pretrain/saved_models/{seq_model}_L.pkl')
            smiles_embedding = load_input_from_pkl(f'../../../pretrain/saved_models/{smi_model}.pkl')
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
