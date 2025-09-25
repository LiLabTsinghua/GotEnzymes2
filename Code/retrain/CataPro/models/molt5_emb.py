import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5EncoderModel
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import MACCSkeys
import re
import pickle
def get_molT5_embed(smiles_list, Molt5_model):
    tokenizer = T5Tokenizer.from_pretrained(Molt5_model)
    model = T5EncoderModel.from_pretrained(Molt5_model)
    N_smiles = len(smiles_list)
    if len(set(smiles_list)) == 1:
        input_ids = tokenizer(smiles_list[0], return_tensors="pt").input_ids
        outputs = model(input_ids=input_ids)
        last_hidden_states = outputs.last_hidden_state
        embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
        final_values = np.concatenate([embed.reshape(1, -1)] * N_smiles, axis=0)
    else:
        final_values = []
        for smile in smiles_list:
            input_ids = tokenizer(smile, return_tensors="pt").input_ids
            outputs = model(input_ids=input_ids)
            last_hidden_states = outputs.last_hidden_state
            embed = torch.mean(last_hidden_states[0][:-1, :], axis=0).detach().cpu().numpy()
            final_values.append(embed.reshape(1, -1))
        final_values = np.concatenate(final_values, axis=0)
    return final_values
def save_embeddings_to_pkl(embeddings, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
df0 = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/CataPro/datasets/kcat-data_0.4simi-10fold.csv')
df1 = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/CataPro/datasets/Km-data_0.4simi-10fold.csv')
df2 = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/CataPro/datasets/kcat-over-Km-data_0.4simi-10fold.csv')
# df2 = pd.read_csv('/home/wuke/project/bio_deeplearning/zzz_benchmark/all_km_models/xiongwen_data/km_all.csv')
# Remove duplicates based on both Sequence and Smiles columns
# df_unique = df.drop_duplicates(subset=['sequence'])

sequence0 = df0['smiles'].to_list()
sequence1 = df1['Smiles'].to_list()
sequence2 = df2['Smiles'].to_list()
sequence = sequence0 + sequence1 + sequence2
# Smiles = ['CCO']

# smiles_to_embedding = defaultdict(list)
seq_to_embedding = defaultdict(list)

# Assuming smiles_to_vec and Seq_to_vec are defined elsewhere and take a list of strings as input.
# unique_smiles_input = list(set(Smiles))
unique_sequence_input = list(set(sequence))
# unique_sequence_input = unique_sequence_input[:10]

# smiles_embeddings = smiles_to_vec(unique_smiles_input)
# for smiles, emb in zip(unique_smiles_input, smiles_embeddings):
#     smiles_to_embedding[smiles] = emb

sequence_embeddings = get_molT5_embed(unique_sequence_input, 'molt5')
for seq, emb in zip(unique_sequence_input, sequence_embeddings):
    seq_to_embedding[seq] = emb

# save_embeddings_to_pkl(smiles_to_embedding, f'smitrans_xiongwen_0414.pkl')

save_embeddings_to_pkl(seq_to_embedding, f'molt5_catapro.pkl')