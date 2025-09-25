import torch
from predict_model import KcatPrediction
import pickle
import pandas as pd
from tqdm import tqdm
model_path = '../../Results/output/' + 'all--radius2--ngram3--dim20--layer_gnn3--window11--layer_cnn3--layer_output3--lr1e-3--lr_decay0.5--decay_interval10--weight_decay1e-6--iteration25'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_tensor(file_name,dtype):
    with open(file_name+'.pkl', 'rb') as f:
        loaded_tensor_list = pickle.load(f)
    return [dtype(d).to(device) for d in loaded_tensor_list]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
dir_input = ('../../Data/input/')
compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)

dataset = list(zip(compounds, adjacencies, proteins, interactions))

model = KcatPrediction().to(device)
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []
true_values = []

with torch.no_grad():
    for compound, adjacency, protein, interaction in tqdm(dataset):
        inputs = (compound, adjacency, protein, interaction)
        predicted_interaction = model(inputs, train=False)  # 设置train为False表示我们是在做预测
        true_values.append(predicted_interaction[0])
        predictions.append(predicted_interaction[1])
df1 = pd.DataFrame(predictions, columns=['Prediction'])
df2 = pd.DataFrame(true_values, columns=['Truth'])
df = pd.concat([df1, df2], axis=1)
df.to_csv('../../Results/predictions.csv', index=False)