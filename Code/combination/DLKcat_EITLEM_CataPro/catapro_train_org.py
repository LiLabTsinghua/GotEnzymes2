from torch import nn
import sys
import re
import torch
from eitlem_utils import Tester, Trainer
# from KCM import EitlemKcatPredictor
# from KMP import EitlemKmPredictor
# from ensemble import ensemble
from cata_model import KcatModel
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

class EitlemDataSet(Dataset):
    def __init__(self, data, sequence_embedding, smiles_embedding1, smiles_embedding2, log10=False):
        super(EitlemDataSet, self).__init__()
        self.data = data
        self.sequence_embedding = sequence_embedding
        self.smiles_embedding1 = smiles_embedding1
        self.smiles_embedding2 = smiles_embedding2
        self.log10 = log10
        # print(f"log10:{self.log10} molType:{self.Type}")

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pro_id = row['Sequence']
        smi_id = row['Smiles']
        value = row['Value']

        protein_emb = self.sequence_embedding[pro_id]
        # print(protein_emb.shape)
        smiles_emb1 = self.smiles_embedding1[smi_id]
        smiles_emb2 = self.smiles_embedding2[smi_id]
        smiles_emb1 = torch.FloatTensor(smiles_emb1)
        smiles_emb2 = torch.FloatTensor(smiles_emb2)
        smiles_emb = torch.cat((smiles_emb1, smiles_emb2), dim=0)
        # print(smiles_emb1.shape, smiles_emb2.shape, smiles_emb.shape)
        if self.log10:
            value = math.log10(value)
        else:
            value = math.log2(value)
        data = Data(x = torch.FloatTensor(smiles_emb).unsqueeze(0), pro_emb=torch.FloatTensor(protein_emb).unsqueeze(0), value=value)
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

def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def kineticsTrainer(kkmPath, TrainType, Type, Iteration, log10, molType, device, smi_model, seq_model, kinetic_parameter):
    for cv in range(5):
    # if cv is not None:
        train_info = f"CATAPRO_Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}-cv{cv}"
        # else:
        #     train_info = f"Transfer-{TrainType}-{Type}-train-{Iteration}-{smi_model}-{seq_model}"

        if os.path.exists(f'../Results/{Type}/{train_info}'):
            return None
        dict1 = {'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 'smitrans': 1024,  'maccskeys': 935, 'chemberta2': 768, 'rdkitfp': 2048, 'unimolv1': 512}
        dict2 = {'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 'prott5': 1024, 'prollama': 4096, 'esmc':1152}
        smi_shape = dict1[smi_model]
        seq_shape = dict2[seq_model]
        csv_path = f'../../../data/cv/{cv}/EITLEM_{kinetic_parameter}.csv'
        data = pd.read_csv(csv_path)
        sequence_embedding = load_input_from_pkl(f'../../../saved_models/{seq_model}.pkl')
        smiles_embedding1 = load_input_from_pkl(f'../../../saved_models/{smi_model}.pkl')
        smiles_embedding2 = load_input_from_pkl(f'../../../saved_models/molt5.pkl')

        train_pair_info, test_pair_info = get_pair_info(data)

        train_set = EitlemDataSet(train_pair_info, sequence_embedding, smiles_embedding1, smiles_embedding2, log10)
        test_set = EitlemDataSet(test_pair_info, sequence_embedding, smiles_embedding1, smiles_embedding2, log10)

        train_loader = EitlemDataLoader(data=train_set, batch_size=200, shuffle=True, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=True)
        valid_loader = EitlemDataLoader(data=test_set, batch_size=200, drop_last=False, num_workers=30, prefetch_factor=10, persistent_workers=True, pin_memory=True)

        Epoch = 100
        
        file_model = f'../Results/{Type}/{train_info}/Weight/'

        model = KcatModel(smi_shape, seq_shape)
        if not os.path.exists(file_model):
            os.makedirs(file_model)
        file_model += 'CataPro_'
        """Train setting."""
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80], gamma=0.9)
        
        loss_fn = nn.MSELoss()
        tester = Tester(device, loss_fn, log10=log10)
        trainer = Trainer(device, loss_fn, log10=log10)
        
        # print("start to training...")
        writer = SummaryWriter(f'../Results/{Type}/{train_info}/logs/')
        for epoch in range(1, Epoch + 1):
            train_MAE, train_rmse, train_r2, loss_train, pcc_train = trainer.run(model, train_loader, optimizer, len(train_pair_info), f"{Iteration} iter epoch {epoch} train:")
            if epoch % 10 == 0:
                MAE_dev, RMSE_dev, R2_dev, loss_dev, pcc_dev = tester.test(model, valid_loader, len(test_pair_info), desc=f"{Iteration} iter epoch {epoch} valid:")
            scheduler.step()
            if epoch % 10 == 0:
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

def TransferLearing(TrainType, log10, molType, device, seq_model, smi_model):
    # smi_model_list = ['maccskeys', 'smitrans', 'molgen', 'molebert', 'unimolv1', 'unimolv2', 'chemberta2', 'ecfp', 'rdkitfp']
    # seq_model_list = ['esm2', 'esm1b', 'esm1v', 'prott5', 'prollama']
    # for smi_model in smi_model_list:
    #     for seq_model in seq_model_list:
    print(f"smi_model: {smi_model}, seq_model: {seq_model}")
    iteration = 1
    kineticsTrainer(None, TrainType, 'KCAT', iteration, log10, molType, device, smi_model, seq_model, 'KCAT')
    kineticsTrainer(None, TrainType, 'KM', iteration, log10, molType, device, smi_model, seq_model, 'KM')
    kineticsTrainer(None, TrainType, 'KKM', iteration, log10, molType, device, smi_model, seq_model, 'KKM')
    torch.cuda.empty_cache()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--TrainType', type=str, required=True)
    parser.add_argument('-l', '--log10', type=bool, default=True)
    parser.add_argument('-m', '--molType', type=str, default='MACCSKeys')
    parser.add_argument('-d', '--device', type=int, required=True)
    parser.add_argument('-seq', type=str, required=True)
    parser.add_argument('-smi', type=str, required=True)
    parser.add_argument('-cv', type=str, default=None)
    return parser.parse_args()

if __name__ == '__main__':
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{args.device}')
    else:
        device = torch.device('cpu')
    print(f"used device {device}")
    TransferLearing(args.TrainType, args.log10, args.molType, device, args.seq, args.smi)
