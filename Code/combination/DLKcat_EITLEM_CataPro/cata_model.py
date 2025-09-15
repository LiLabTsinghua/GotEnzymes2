import torch as th
import torch.nn as nn

class KcatModel(nn.Module):
    def __init__(self, mol_in_dim, protein_dim, rate=0.0, device="cuda:0"):
        super(KcatModel, self).__init__()
        self.mol_in_dim = mol_in_dim
        self.protein_dim = protein_dim
        self.prot_norm = nn.BatchNorm1d(protein_dim).to(device)
        self.molt5_norm = nn.BatchNorm1d(mol_in_dim).to(device)
        self.decoder = nn.Sequential(nn.Linear(mol_in_dim + protein_dim, 256), nn.BatchNorm1d(256), nn.Dropout(p=rate), nn.ReLU(),).to(device)

        self.out = nn.Sequential(nn.Linear(256, 1)).to(device)

    def forward(self, data):
        ezy_feats = data.pro_emb
        sbt_feats = data.x
        # print("ezy_feats.shape", ezy_feats.shape)
        # print("sbt_feats.shape", sbt_feats.shape)
        prot_feats = self.prot_norm(ezy_feats)
        molt5_feats = self.molt5_norm(sbt_feats)
        # macc_feats = sbt_feats[:, self.mol_in_dim:]
        cplx_feats = th.cat([prot_feats, molt5_feats], axis=1)

        feats = self.decoder(cplx_feats) 

        out = self.out(feats)

        return out.squeeze(-1)

# class KmModel(nn.Module):
#     def __init__(self, mol_in_dim, protein_dim, rate=0.0, device="cuda:0"):
#         super(KmModel, self).__init__()
#         self.mol_in_dim = mol_in_dim
#         self.protein_dim = protein_dim
#         self.prot_norm = nn.BatchNorm1d(protein_dim).to(device)
#         self.molt5_norm = nn.BatchNorm1d(mol_in_dim).to(device)
#         self.decoder = nn.Sequential(nn.Linear(protein_dim + mol_in_dim, 256), nn.BatchNorm1d(256), nn.Dropout(p=rate), nn.ReLU(),).to(device)

#         self.out = nn.Sequential(nn.Linear(256, 1)).to(device)

#     def forward(self, data):
#         ezy_feats = data.pro_emb
#         sbt_feats = data.x
#         prot_feats = self.prot_norm(ezy_feats)
#         molt5_feats = self.molt5_norm(sbt_feats)
#         # macc_feats = sbt_feats[:, self.mol_in_dim:]
#         cplx_feats = th.cat([prot_feats, molt5_feats], axis=1)

#         feats = self.decoder(cplx_feats)

#         out = self.out(feats)

#         return out


# class ActivityModel(nn.Module):
#     def __init__(self, mol_in_dim, protein_dim, rate=0.0, alpha=0.4, device="cuda:0"):
#         super(ActivityModel, self).__init__()
#         self.mol_in_dim = mol_in_dim
#         self.protein_dim = protein_dim
#         self.alpha = alpha

#         self.kcat_model = KcatModel().to(device)
#         self.Km_model = KmModel().to(device)

#         self.prot_norm = nn.BatchNorm1d(self.protein_dim).to(device)
#         self.molt5_norm = nn.BatchNorm1d(self.mol_in_dim).to(device)

#         self.decoder = nn.Sequential(nn.Linear(1959, 256), nn.BatchNorm1d(256), nn.Dropout(p=rate), nn.ReLU()).to(device)
#         self.attn = nn.Sequential(nn.Linear(256, 256), nn.Softmax(dim=1)).to(device)
#         self.out = nn.Linear(256, 1).to(device)

#     def forward(self, data):
#         ezy_feats = data.pro_emb
#         sbt_feats = data.x
#         pred_kcat, _ = self.kcat_model(ezy_feats, sbt_feats)
#         pred_Km, _ = self.Km_model(ezy_feats, sbt_feats)

#         #out = th.log10(th.pow(10, pred_kcat) / th.pow(10, pred_Km))
#         pred_activity_1 = pred_kcat - pred_Km  # log(kcat/Km) = log(kcat) - log(Km)

#         ezy_feats = self.prot_norm(ezy_feats)
#         molt5_feats = self.molt5_norm(sbt_feats)
#         # macc_feats = sbt_feats[:, self.mol_in_dim:]
#         cplx_feats = th.cat([ezy_feats, molt5_feats], axis=1)
#         feats = self.decoder(cplx_feats)
#         attn_score = self.attn(feats)
#         attn_feats = attn_score * feats

#         pred_activity_2 = self.out(attn_feats)
#         pred_activity = pred_activity_1.detach() * (1 - self.alpha) + pred_activity_2 * self.alpha

#         return pred_kcat, pred_Km, pred_activity
