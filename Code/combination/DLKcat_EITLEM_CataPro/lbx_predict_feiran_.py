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
# '/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/EITLEM_Kinetics/Results/KCAT/Transfer-240121-KCAT-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8815_devR2_0.6169_RMSE_0.9454_MAE_0.6401'
modelPath = {
    'KCAT':'../KCAT/iter8_trainR2_0.9408_devR2_0.7459_RMSE_0.7751_MAE_0.4787',
    'KM':'../KM/iter8_trainR2_0.9303_devR2_0.7163_RMSE_0.6960_MAE_0.4802',
    'KKM':'../KKM/iter8-trainR2_0.9091_devR2_0.8325_RMSE_0.7417_MAE_0.4896'
}
# Load ESM1v model
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
batch_converter = alphabet.get_batch_converter()
model.eval()
kinetics_type = 'KCAT'
if kinetics_type == 'KCAT':
    eitlem = EitlemKcatPredictor(167, 512, 1280, 10, 0.5, 10)
elif kinetics_type == 'KM':
    eitlem = EitlemKmPredictor(167, 512, 1280, 10, 0.5, 10)
else:
    eitlem = ensemble(167, 512, 1280, 10, 0.5, 10)
eitlem.load_state_dict(torch.load(modelPath[kinetics_type]))
eitlem.eval()

def predict(eitlem, sequence, smiles):
    # Extratc protein representation
    data = [
    ("protein1", sequence),
    ]
    _, _, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1])
    # Compute the MACCS Keys of substrate
    mol = Chem.MolFromSmiles(smiles)
    mol_feature = MACCSkeys.GenMACCSKeys(mol).ToList()

    sample = Data(x = torch.FloatTensor(mol_feature).unsqueeze(0), pro_emb=sequence_representations[0])
    input_data = Batch.from_data_list([sample], follow_batch=['pro_emb'])

    # Predict kinetics value.
    with torch.no_grad():
        res = eitlem(input_data)
    return math.pow(10,res[0].item())

df = pd.read_csv('/home/wuke/project/bio_deeplearning/kcatkm_predict/result/酶的kcat结果预测4.csv')
results = []
for index, row in tqdm(df.iterrows()):
    kinetics_type = 'KCAT'
    sequence = row['Sequence']
    if len(sequence) > 1022:
        sequence = sequence[:1022]
    smiles = row['Smiles']
    prediction = predict(eitlem, sequence, smiles)
    results.append(prediction)
df['EITLEM_Prediction'] = results
df.to_csv('/home/wuke/project/bio_deeplearning/kcatkm_predict/result/酶的kcat结果预测5.csv', index=False)