import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import degree
from torch_geometric.utils import softmax
from torch_geometric.nn import GCNConv
import pickle
def load_input_from_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)
class Resnet(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, out_dim)
        self.lin2 = nn.Linear(out_dim, out_dim)
        self.lin3 = nn.Linear(out_dim, out_dim)
        self.lin4 = nn.Linear(out_dim, out_dim)

    def forward(self, x):
        residual = F.relu(self.lin1(x))
        for lin in [self.lin2, self.lin3, self.lin4]:
            residual = residual + F.relu(lin(residual))
        return residual

class ProMolAtt(nn.Module):
    def __init__(self, hidden_dim):
        super(ProMolAtt, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.merge = nn.Linear(2*hidden_dim, 1, bias=False) # 计算相似性函数
        self.k = nn.Linear(hidden_dim, hidden_dim)
    def forward(self, mol, prot, batch):
        q = F.relu(self.q(mol)) # 分子映射
        r = q.repeat_interleave(degree(batch,  dtype=batch.dtype), dim=0) # 分子扩增
        k = F.relu(self.k(prot))
        # print(k.shape, r.shape)
        score = self.merge(torch.cat([k, r], dim=-1)) # 计算相似性分数
        score = softmax(score, batch, dim=0) # 权重加权
        o = global_add_pool(k * score, batch) # 聚合全局向量
        return o, q

class AttentionAgg(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionAgg, self).__init__()
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x, y):
        """
        x -> y ==> y^
        """
        q = F.relu(self.q(x.mean(dim=1)))
        k = F.relu(self.k(y))
        score = F.softmax(torch.matmul(q.unsqueeze(1), k.transpose(-1, -2)), dim=-1)
        out = torch.matmul(score, y).squeeze(1)
        return out

class MultiHeadAttenAgg(nn.Module):
    def __init__(self, hidden_dim, att_layer, dropout):
        super().__init__()
        self.seq_m = nn.ModuleList([AttentionAgg(hidden_dim) for _ in range(att_layer)])
        self.seq_o = nn.Sequential(
            nn.Linear(hidden_dim * att_layer, 4 * hidden_dim * att_layer),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * hidden_dim * att_layer, hidden_dim)
        )

    def forward(self, x, y):
        return self.seq_o(torch.cat([m(x, y) for m in self.seq_m], dim=-1))

class ProteinCNN(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=128):
        super().__init__()
        # 多尺度卷积层
        self.conv_branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.3)
            ),
            nn.Sequential(
                nn.Conv1d(embed_dim, hidden_dim, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim),
                nn.PReLU(),
                nn.Dropout(0.3))
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),  # 将多尺度特征融合为 256 维
            nn.PReLU(),
            nn.Dropout(0.3))
    
    def forward(self, x, batch):
        """
        x: (total_seq_len, embed_dim)
        batch: (total_seq_len,) 表示每个位置所属的样本索引
        """
        # Step 1: 分割序列
        lengths = degree(batch, dtype=torch.long)
        max_len = lengths.max()
        
        # Step 2: 验证 lengths 的合法性
        total_length = x.size(0)
        assert sum(lengths) == total_length, "sum(lengths) must equal total_seq_len"
        
        # Step 3: 转换为批处理格式
        batch_size = len(lengths)
        padded = torch.zeros(batch_size, max_len, x.size(1), device=x.device)
        mask = torch.zeros(batch_size, max_len, device=x.device)
        
        # Step 4: 填充序列并创建mask
        cum_length = 0
        for i in range(batch_size):
            seq_len = lengths[i]
            padded[i, :seq_len] = x[cum_length:cum_length+seq_len]
            mask[i, :seq_len] = 1.0
            cum_length += seq_len
        
        # Step 5: 卷积处理
        x = padded.permute(0, 2, 1)  # (B, C, L)
        conv_outputs = []
        for conv in self.conv_branches:
            out = conv(x)  # (B, H, L)
            conv_outputs.append(out.permute(0, 2, 1))  # (B, L, H)
        
        # Step 6: 多尺度特征融合
        combined = torch.cat(conv_outputs, dim=-1)  # (B, L, 2H)
        
        # Step 7: 特征融合为 256 维
        fused_features = self.feature_fusion(combined)  # (B, L, 256)
        
        # Step 8: 还原为 total_seq_len 格式
        output = torch.zeros(total_length, 256, device=x.device)
        cum_length = 0
        for i in range(batch_size):
            seq_len = lengths[i]
            # 确保 seq_len 不超过 fused_features 的实际长度
            seq_len = min(seq_len, fused_features.size(1))
            output[cum_length:cum_length+seq_len] = fused_features[i, :seq_len]
            cum_length += seq_len
        
        return output

class EitlemKcatPredictor(nn.Module):
    def __init__(self, 
                 mol_in_dim, 
                 hidden_dim=128, 
                 protein_dim=1280, 
                 layer=10, 
                 dropout=0.2, 
                 att_layer=10
                ):
        super(EitlemKcatPredictor, self).__init__()
        self.prej1 = Resnet(mol_in_dim, hidden_dim)
        self.prej2 = nn.Linear(protein_dim, hidden_dim, bias=False)
        self.pro_extrac = nn.ModuleList([ProMolAtt(hidden_dim) for _ in range(layer)])
        self.att1 = MultiHeadAttenAgg(hidden_dim, att_layer, dropout)
        self.att2 = MultiHeadAttenAgg(hidden_dim, att_layer, dropout)
        self.out = nn.Sequential(
            nn.Linear(2 * hidden_dim, 4 * hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.PReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 1)
        )
        # CNN model:
        n_word = len(load_input_from_pickle(f'../Data/CNNGNN/sequence_dict_KKM.pickle'))
        self.embed_word = nn.Embedding(n_word, 128)
        # self.W_cnn = nn.ModuleList([nn.Conv1d(
        #     in_channels=128, out_channels=128, kernel_size=3, padding=1) for _ in range(3)])
        self.W_cnn = ProteinCNN(embed_dim=128, hidden_dim=128)
        
        # GNN model:
        n_fingerprint = len(load_input_from_pickle(f'../Data/CNNGNN/fingerprint_dict_KKM.pickle'))
        self.embed_fingerprint = nn.Embedding(n_fingerprint, 512)
        self.conv1 = GCNConv(512, 512)
        self.conv2 = GCNConv(512, 128)
        
    def attention_cnn(self, xs, batch):
        return self.W_cnn(xs, batch)
    
    def final_stage(self, mol, pro):
        pro_out = self.att1(mol, pro)
        mol_out = self.att2(pro, mol)
        return self.out(torch.cat([mol_out, pro_out], dim=-1)).squeeze(dim=-1)

    def forward(self, data):
        # print(data.use_gnn, data.use_cnn)
        if isinstance(data.use_gnn, torch.Tensor):
            use_gnn = data.use_gnn.all().item()
        else:
            use_gnn = data.use_gnn
        
        if isinstance(data.use_cnn, torch.Tensor):
            use_cnn = data.use_cnn.all().item()
        else:
            use_cnn = data.use_cnn

        if use_gnn:
            fingerprints= self.embed_fingerprint(data.x)
            adjacency = data.edge_index
            x = self.conv1(fingerprints, adjacency)
            x = F.relu(x)
            x = self.conv2(x, adjacency)
            smi = global_add_pool(x, data.batch)
        else:
            smi = data.x
        if use_cnn:
            word_vector = self.embed_word(data.pro_emb)
            # print('word_vector.shape', word_vector.shape)
            # print('word_vector', word_vector)
            seq = self.attention_cnn(word_vector, data.pro_emb_batch)
        else:
            seq = data.pro_emb
        # print('smi.shape seq.shape', smi.shape, seq.shape)
        # print('data.pro_emb_batch',data.pro_emb_batch)
        mol = F.relu(self.prej1(smi))
        prot = F.relu(self.prej2(seq))
        att_pro = []
        att_mol = []
        for m in self.pro_extrac:
            o, q = m(mol, prot, data.pro_emb_batch)
            att_pro.append(o)
            att_mol.append(q)
        att_mol = torch.stack(att_mol, dim=1)
        att_pro = torch.stack(att_pro, dim=1)
        return self.final_stage(att_mol, att_pro)
