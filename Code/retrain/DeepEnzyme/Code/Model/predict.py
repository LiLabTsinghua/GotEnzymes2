import torch
from DeepEnzyme import DeepEnzyme
import pickle
import pandas as pd
from tqdm import tqdm
import numpy as np
model_path = '../../Results/Output/dim64_lr001_E200_head4_drop3_seed666'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
dir_input = '../../Data/Input/'
fingerprint = load_tensor(dir_input + 'fingerprint', torch.LongTensor)
smileadjacencies = load_tensor(dir_input + 'smilesadjacencies', torch.FloatTensor)
sequences = load_tensor(dir_input + 'sequences', torch.LongTensor)
with open(dir_input + 'proteinadjacencies.pkl', 'rb') as f:
    proteinadjacencies = pickle.load(f)
kcat_tensor = load_tensor(dir_input + 'regression', torch.FloatTensor)
# print(len(fingerprint), len(smileadjacencies), len(sequences), len(proteinadjacencies), len(kcat_tensor))
# print(sequences[1].shape, proteinadjacencies[1].shape)
# dict
fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
word_dict = load_pickle(dir_input + 'word_dict.pickle')
n_fingerprint = len(fingerprint_dict)
n_word = len(word_dict)
lr = 0.001
iteration = 200
weight_decay = 1e-6
dropout = 0.3
dim = 64
layer_output = 3
hidden_dim1 = 64
hidden_dim2 = 64
nhead = 4
hid_size = 64
layers_trans = 3
dataset = list(zip(fingerprint, smileadjacencies, sequences, proteinadjacencies, kcat_tensor))

model = DeepEnzyme(n_fingerprint, dim, n_word, layer_output, hidden_dim1, hidden_dim2, dropout, nhead, hid_size,
                   layers_trans).to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []
true_values = []

with torch.no_grad():
    for fingerprint, smileadjacency, sequence, proteinadjacency, kcat in tqdm(dataset):
        inputs = (fingerprint, smileadjacency, sequence, proteinadjacency, None)
        predicted_interaction = model(inputs, layer_output=layer_output, dropout=dropout)
        predictions.append(predicted_interaction.item())
        true_values.append(kcat.item())

df_predictions = pd.DataFrame(predictions, columns=['Prediction'])
df_true_values = pd.DataFrame(true_values, columns=['Truth'])
df = pd.concat([df_predictions, df_true_values], axis=1)
df.to_csv('../../Results/predictions.csv', index=False)