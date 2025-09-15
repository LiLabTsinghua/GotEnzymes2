python predict_250216.py \
    --csv_path /home/wuke/project/bio_deeplearning/zzz_benchmark/results_250216/KCAT_sml.csv \
    --smiles_pkl_path ../../../pretrain/saved_models/maccskeys.pkl \
    --sequence_pkl_path ../../../pretrain/saved_models/esm2_L.pkl \
    --model_path ../Results/KCAT/Transfer-240121-KCAT-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8815_devR2_0.6169_RMSE_0.9454_MAE_0.6401 \
    --kinetics_type KCAT \
    --device 0
# python predict_250216.py \
#     --csv_path /home/wuke/project/bio_deeplearning/zzz_benchmark/results_250216/KM_sml.csv \
#     --smiles_pkl_path ../../../pretrain/saved_models/maccskeys.pkl \
#     --sequence_pkl_path ../../../pretrain/saved_models/esm2_L.pkl \
#     --model_path ../Results/KM/Transfer-240121-KM-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2_0.8780_devR2_0.5600_RMSE_0.8384_MAE_0.6090 \
#     --kinetics_type KM \
#     --device 0
# python predict_250216.py \
#     --csv_path /home/wuke/project/bio_deeplearning/zzz_benchmark/results_250216/KKM_sml.csv \
#     --smiles_pkl_path ../../../pretrain/saved_models/maccskeys.pkl \
#     --sequence_pkl_path ../../../pretrain/saved_models/esm2_L.pkl \
#     --model_path ../Results/KKM/Transfer-240121-KKM-train-1-maccskeys-esm2/Weight/Eitlem_MACCSKeys_trainR2:0.8999_devR2_0.5510_RMSE_1.1698_MAE_0.8283 \
#     --kinetics_type KKM \
#     --device 0