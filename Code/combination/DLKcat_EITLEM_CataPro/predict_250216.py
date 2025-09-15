import torch
import pandas as pd
import pickle
from torch_geometric.data import Batch, Data
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
import math
import argparse
import numpy as np

class EitlemDataSet:
    def __init__(self, data, sequence_embedding, smiles_embedding):
        self.data = data
        self.sequence_embedding = sequence_embedding
        self.smiles_embedding = smiles_embedding

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        pro_id = row['Sequence']
        smi_id = row['Smiles']

        protein_emb = self.sequence_embedding[pro_id]
        smiles_emb = self.smiles_embedding[smi_id]

        data = Data(x=torch.FloatTensor(smiles_emb).unsqueeze(0), pro_emb=torch.FloatTensor(protein_emb))
        return data

    def collate_fn(self, batch):
        return Batch.from_data_list(batch, follow_batch=['pro_emb'])

    def __len__(self):
        return len(self.data)

def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def predict_csv(csv_path, smiles_pkl_path, sequence_pkl_path, model_path, kinetics_type, device):
    # 加载数据
    data = pd.read_csv(csv_path)
    if 'Prediction' in data.columns:
        data.drop(columns=['Prediction'], inplace=True)
    sequence_embedding = load_input_from_pkl(sequence_pkl_path)
    smiles_embedding = load_input_from_pkl(smiles_pkl_path)

    # 创建数据集
    dataset = EitlemDataSet(data, sequence_embedding, smiles_embedding)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=200, collate_fn=dataset.collate_fn)

    # 加载模型
    if kinetics_type == 'KCAT':
        model = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
    elif kinetics_type == 'KM':
        model = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
    elif kinetics_type == 'KKM':
        model = ensemble(167, 512, 1280, 10, 0.5, 10)
    else:
        raise ValueError("Invalid kinetics_type. Must be 'KCAT', 'KM', or 'KKM'.")

    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    # 预测
    predictions = []
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().numpy())

    # 将预测结果添加到 DataFrame
    data[f'Prediction'] = np.power(10, predictions)

    # 保存结果
    output_csv_path = csv_path.replace('.csv', f'_{kinetics_type}_prediction.csv')
    data.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to {output_csv_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--smiles_pkl_path', type=str, required=True, help='Path to the SMILES embedding pkl file')
    parser.add_argument('--sequence_pkl_path', type=str, required=True, help='Path to the Sequence embedding pkl file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--kinetics_type', type=str, required=True, choices=['KCAT', 'KM', 'KKM'], help='Type of kinetics (KCAT, KM, or KKM)')
    parser.add_argument('--device', type=int, default=0, help='Device to use for prediction (e.g., 0 for GPU)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    predict_csv(
        csv_path=args.csv_path,
        smiles_pkl_path=args.smiles_pkl_path,
        sequence_pkl_path=args.sequence_pkl_path,
        model_path=args.model_path,
        kinetics_type=args.kinetics_type,
        device=device
    )
'''
python predict.py \
    --csv_path /home/wuke/project/bio_deeplearning/zzz_benchmark/results_250216/DLKcat_prediction.csv \
    --smiles_pkl_path ../../../pretrain/saved_models/maccskeys.pkl \
    --sequence_pkl_path ../../../pretrain/saved_models/esm2_L.pkl \
    --model_path ../Results/KCAT/Transfer-240121-KCAT-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8815_devR2_0.6169_RMSE_0.9454_MAE_0.6401 \
    --kinetics_type KCAT \
    --device 0
'''
