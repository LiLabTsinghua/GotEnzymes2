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


class KcatModel(nn.Module):
    def __init__(self, mol_in_dim, protein_dim):
        super(KcatModel, self).__init__()

        self.W_attention = nn.Linear(20, 20)
        self.W_out = nn.ModuleList([nn.Linear(40, 40)
                                    for _ in range(3)])
        self.W_interaction = nn.Linear(40, 1)
        self.prot_emb = nn.Linear(protein_dim, 20)
        self.mol_emb = nn.Linear(mol_in_dim, 20)
    def attention_cnn(self, x, xs):
        """The attention mechanism is applied to the last layer of CNN."""
        h = torch.relu(self.W_attention(x))  # [200, 20]
        hs = torch.relu(self.W_attention(xs))  # [200, 20]

        # Compute attention scores (dot product)
        scores = torch.matmul(h, hs.T)  # [200, 200]

        # Apply softmax to get attention weights
        weights = torch.softmax(scores, dim=1)  # [200, 200]

        # Weighted sum of hs
        ys = torch.matmul(weights, hs)  # [200, 20]

        return ys


    def forward(self, data):
        # print("data.pro_emb.shape", data.pro_emb.shape)
        # print("data.x.shape", data.x.shape)
        compound_vector = self.prot_emb(data.pro_emb)
        word_vectors = self.mol_emb(data.x)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors)
        # print("compound_vector.shape", compound_vector.shape)
        # print("protein_vector.shape", protein_vector.shape)
        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(3):
            cat_vector = torch.relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        return interaction.squeeze(-1)
