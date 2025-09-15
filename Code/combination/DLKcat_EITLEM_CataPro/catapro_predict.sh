km='KKM'
python catapro_predict.py \
    --input_csv ../../../data/cv/0/EITLEM_$km.csv \
    --model_path ../Results/$km/CATAPRO_Transfer-catapro-$km-train-1-maccskeys-prott5-cv0/Weight/CataPro_MACCSKeys_trainR2_0.8400_devR2_0.5328_RMSE_1.1934_MAE_0.8541_PCC_0.7426 \
    --output_csv ../catapro_results/$km.csv \
    --smi_model maccskeys \
    --seq_model prott5 \
    --log10