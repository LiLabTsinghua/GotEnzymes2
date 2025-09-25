#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1

export CUDA_VISIBLE_DEVICES=1
cd /home/wuke/project/bio_deeplearning/zzz_benchmark/all_opt_models/Seq2Topt/code

start_time=$(date +%s)
python run_train_new.py --task tm --train_path ../data/Tm/Tm_new_Train.csv --test_path ../data/Tm/Tm_new_Test.csv --seq_model esm1b --lr 0.0001 --cv 0
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed "time"
echo "Tm Done!"
echo "time used" $elapsed "seconds"
