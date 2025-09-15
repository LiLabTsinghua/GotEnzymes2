import torch
import pandas as pd
from torch import nn
from dataset import EitlemDataLoader
from cata_model import KcatModel
import pickle
import math
import argparse
import os
from torch_geometric.data import Batch, Dataset, Data

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
    
class Predictor:
    def __init__(self, device, log10=False):
        self.device = device
        self.log10 = log10
        
    def load_model(self, model_path, smi_shape, seq_shape):
        model = KcatModel(smi_shape, seq_shape)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()
        return model
    
    def predict(self, model, data_loader):
        predictions = []
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                output = model(batch)
                
                if self.log10:
                    output = torch.pow(10, output)
                else:
                    output = torch.pow(2, output)
                
                predictions.extend(output.squeeze().cpu().numpy())
        return predictions

def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output_csv', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--smi_model', type=str, required=True, help='SMILES embedding model name')
    parser.add_argument('--seq_model', type=str, required=True, help='Sequence embedding model name')
    parser.add_argument('--log10', action='store_true', help='Whether to use log10 scaling')
    parser.add_argument('--device', type=int, default=0, help='CUDA device number')
    args = parser.parse_args()

    # Define embedding dimensions
    smi_dims = {
        'molgen': 1024, 'unimolv2': 1024, 'molebert': 300, 'ecfp': 1024, 
        'smitrans': 1024, 'maccskeys': 935, 'chemberta2': 768, 
        'rdkitfp': 2048, 'unimolv1': 512
    }
    seq_dims = {
        'esm2': 1280, 'esm1b': 1280, 'esm3b': 2560, 'esm15b': 5120, 
        'prott5': 1024, 'prollama': 4096, 'esmc': 1152
    }

    # Set up device
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # Load data
    data = pd.read_csv(args.input_csv)
    
    # Load embeddings
    sequence_embedding = load_input_from_pkl(f'../../../saved_models/{args.seq_model}.pkl')
    smiles_embedding1 = load_input_from_pkl(f'../../../saved_models/{args.smi_model}.pkl')
    smiles_embedding2 = load_input_from_pkl(f'../../../saved_models/molt5.pkl')

    # Create dataset and dataloader
    dataset = EitlemDataSet(data, sequence_embedding, smiles_embedding1, smiles_embedding2, args.log10)
    data_loader = EitlemDataLoader(
        data=dataset, 
        batch_size=200, 
        shuffle=False, 
        drop_last=False, 
        num_workers=4
    )

    # Initialize predictor
    predictor = Predictor(device, args.log10)
    
    # Load model
    model = predictor.load_model(
        args.model_path, 
        smi_dims[args.smi_model], 
        seq_dims[args.seq_model]
    )

    # Make predictions
    predictions = predictor.predict(model, data_loader)

    # Save results
    result_df = data.copy()
    result_df['prediction'] = predictions
    result_df.to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == '__main__':
    main()