#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU
#SBATCH --time=10:00:00 
#SBATCH --partition=short
#SBATCH --gres=gpu:rtx8000:1

export CUDA_VISIBLE_DEVICES=3

echo "Hyp-params opt!"
start_time=$(date +%s)

python run_train_new.py --task topt --train_path ../data/Topt/train_os.csv --test_path ../data/Topt/test.csv --seq_model prott5 --lr 0.0001 --cv 4

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "time used" $elapsed "seconds"


start_time=$(date +%s)
python run_train_new.py --task tm --train_path ../data/Tm/Tm50_Train.csv --test_path ../data/Tm/Tm50_Test.csv --seq_model prott5 --lr 0.0001 --cv 4
end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo $elapsed "time"
echo "Tm Done!"
echo "time used" $elapsed "seconds"
