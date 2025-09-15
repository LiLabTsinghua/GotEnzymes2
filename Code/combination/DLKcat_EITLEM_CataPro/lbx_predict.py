import torch
import esm
import sys
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from torch_geometric.data import Data, Batch
import math
sys.path.append("../Code/")
from KCM import EitlemKcatPredictor
from KMP import EitlemKmPredictor
from ensemble import ensemble
import pandas as pd
from tqdm import tqdm
modelPath = {
    'KCAT':'/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/EITLEM_Kinetics/Results/KCAT/Transfer-240121-KCAT-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8815_devR2_0.6169_RMSE_0.9454_MAE_0.6401',
    'KM':'/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/EITLEM_Kinetics/Results/KM/Transfer-240121-KM-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8780_devR2_0.5600_RMSE_0.8384_MAE_0.6090',
    'KKM':'/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/EITLEM_Kinetics/Results/KKM/Transfer-240121-KKM-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2:0.8999_devR2_0.5510_RMSE_1.1698_MAE_0.8283'
}
# modelPath = {
#     'KCAT':'../KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787',
#     'KM':'../KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802',
#     'KKM':'../KKM/iter8-trainR2_0.9091_devR2_0.8325_RMSE_0.7417_MAE_0.4896'
# }

model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()

def predict_dataframe(df):
    predictions = []

    for index, row in tqdm(df.iterrows()):
        kinetics_type = 'KCAT'
        sequence = row['Sequence']
        smiles = row['Smiles']

        # Extract protein representation
        data = [("protein1", sequence)]
        _, _, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        sequence_representation = token_representations[0, 1 : batch_lens[0] - 1]

        # Compute the MACCS Keys of substrate
        mol = Chem.MolFromSmiles(smiles)
        mol_feature = MACCSkeys.GenMACCSKeys(mol).ToList()
        # print(len(mol_feature))
        # print(sequence_representation.shape)
        sample = Data(x=torch.FloatTensor(mol_feature).unsqueeze(0), pro_emb=sequence_representation.unsqueeze(0))
        input_data = Batch.from_data_list([sample], follow_batch=['pro_emb'])
        print(input_data.pro_emb_batch)
        # Initialize predictor based on kinetics type
        if kinetics_type == 'KCAT':
            eitlem = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
        elif kinetics_type == 'KM':
            eitlem = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
        else:
            eitlem = ensemble(167, 512, 1280, 10, 0.5, 10)

        eitlem.load_state_dict(torch.load(modelPath[kinetics_type]))
        eitlem.eval()

        # Predict kinetics value.
        with torch.no_grad():
            res = eitlem(input_data)
        prediction = math.pow(10, res[0].item())
        predictions.append(prediction)

    # Add predictions to a new column in the DataFrame
    df['EITLEM_Prediction'] = predictions
    
    return df

data_df = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/results_250216/DLKcat_prediction.csv')
# data_df = data_df[data_df['Test'] == 1]
predicted_df = predict_dataframe(data_df)
predicted_df.to_csv('../Results/酶的kcat结果预测5.csv', index=False)
# print(predicted_df)
