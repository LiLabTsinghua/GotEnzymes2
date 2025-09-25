#!/usr/bin/python
# coding: utf-8

# Author: LE YUAN
# Date: 2020-10-23
import os
import pickle
import sys
import timeit
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
import warnings
warnings.filterwarnings("ignore")
class KcatPrediction(nn.Module):
    def __init__(self):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = torch.relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
#           xs = torch.relu(self.W_cnn[i](xs))
            xs = F.leaky_relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = torch.relu(self.W_attention(x))
        hs = torch.relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        # print(xs.shape, h.shape, hs.shape, weights.shape, ys.shape)
        # torch.Size([1479, 20]) torch.Size([1, 20]) torch.Size([1479, 20]) torch.Size([1, 1479]) torch.Size([1479, 20])
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs
        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_interaction[0], correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []
        for data in dataset:
            loss, correct_values, predicted_values = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            trainCorrect.append(correct_values)
            trainPredict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict))
        r2_train = r2_score(trainCorrect,trainPredict)
        return loss_total, rmse_train, r2_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        return MAE, rmse, r2

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

# def load_tensor(file_name, dtype):
#     return [dtype(d).to(device) for d in np.load(file_name + '.pkl', allow_pickle=True)]###
# def load_tensor(file_name, dtype):
#     return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]#原来的
def load_tensor(file_name,dtype):
    with open(file_name+'.pkl', 'rb') as f:
        loaded_tensor_list = pickle.load(f)
    return [dtype(d).to(device) for d in loaded_tensor_list]


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


def run_with_os_system(script_path, *args):
    # 需要手动将参数转义，如果它们包含特殊字符，这可能很棘手且不安全
    # 这里为了简单，假设参数不含特殊字符
    args_str = " ".join(args)
    command = f"{sys.executable} {script_path} {args_str}"
    print(f"Running command with os.system: {command}")

    # os.system() 会将命令传递给系统的 shell 执行
    # 返回值通常是 shell 的退出状态，而不是脚本的直接退出码（在某些系统上可能相同）
    # 标准输出和错误会直接打印到当前终端，无法方便地捕获
    exit_status = os.system(command)

    print(f"os.system returned: {exit_status}")
    if exit_status == 0:
        print("Script seems to have executed successfully (based on os.system return).")
    else:
        print("Script execution might have failed.")

if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    # print(type(radius))
    
    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    # def load_data(dir_input):
    dir_input = ('../../Data/xw_input_kkm/train_data/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    fingerprint_dict = load_pickle('../../Data/xw_input_kkm/dictionaries/fingerprint_dict.pickle')
    word_dict = load_pickle('../../Data/xw_input_kkm/dictionaries/sequence_dict.pickle')
    dataset_ = list(zip(compounds, adjacencies, proteins, interactions))
    n_fingerprint= len(fingerprint_dict)
    n_word = len(word_dict)
    
    dir_input = ('../../Data/xw_input_kkm/test_data/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    dataset_test = list(zip(compounds, adjacencies, proteins, interactions))
    # return compounds, adjacencies, proteins, interactions, n_fingerprint, n_word
    # dataset_traindev = load_data('../../Data/xw_input_train/')
    # dataset_test = load_data('../../Data/xw_input_test/')
    
    """Create a dataset and split it into train,dev,test=0.8,0.1,0.1."""
    # dataset = [(compound, adjacency, protein, interaction + 10) for compound, adjacency, protein, interaction in dataset]
    # dataset = shuffle_dataset(dataset, 678)
    # print(len(dataset))
    dataset_train, dataset_dev = split_dataset(dataset_, 0.9)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.5)
    dataset_all = dataset_ + dataset_test
    """Set a model."""
    seed_number = 1234
    print(f'{seed_number} seed_number')
    torch.manual_seed(seed_number)
    model = KcatPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)
    
    """Output files."""
    file_MAEs = '../../Results/output/xwkkmMAEs--' + setting + '.txt'
    file_model = '../../Results/output/xwkkm' + setting
    print(file_MAEs)
    print(file_model)
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test')
    import os
    print(os.path.exists(file_MAEs))
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')
    
    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        # MAEs = [epoch, time, rmse_train, r2_train, MAE_dev,
        #         MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test]
        MAEs = [epoch, round(time, 4), round(rmse_train, 4),
                round(r2_train, 4),  round(MAE_dev, 4), round(MAE_test, 4), round(RMSE_dev, 4),
                round(RMSE_test, 4), round(R2_dev, 4), round(R2_test, 4)]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
    """Save predictions for the entire training set."""
    print('Saving predictions for the training set...')
    train_predictions_file = '../../Results/output/xw_kkm_predictions' + '.txt'
    with open(train_predictions_file, 'w') as f:
        f.write('Correct\tPredicted\n')
        for data in dataset_all:
            correct_values, predicted_values = model(data, train=False)
            correct_values = math.log10(math.pow(2, correct_values))
            predicted_values = math.log10(math.pow(2, predicted_values))
            f.write(f'{correct_values}\t{predicted_values}\n')
    print(f'Training set predictions saved to {train_predictions_file}')