#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --job-name=DL_GPU_Parallel_Fixed
#SBATCH --time=10:00:00 
#SBATCH --partition=short

# --- 并行配置 ---
SEQ_MODELS=("esmc")
CV_FOLDS=(3 4)
MAX_JOBS=5

# --- 环境设置 ---
# 进入工作目录
cd /home/wuke/project/bio_deeplearning/zzz_benchmark/all_opt_models/Seq2Topt/code
# SLURM会自动设置CUDA_VISIBLE_DEVICES，我们不需要也不应该在循环中修改它。
# 所有子进程都会继承这个设置，从而使用同一个GPU。

# --- 主逻辑 ---
echo "开始并行运行，最大并发数: $MAX_JOBS"
start_time=$(date +%s)

# 使用双重循环生成所有参数组合
for model in "${SEQ_MODELS[@]}"; do
    for cv in "${CV_FOLDS[@]}"; do
        # 检查当前后台任务数量是否已达到上限
        while (( $(jobs -p | wc -l) >= MAX_JOBS )); do
            wait -n
        done

        echo "启动任务: model=${model}, cv=${cv}"
        
        # 将 Python 命令放到后台执行
        # 2. 在命令末尾添加了 & 使其后台运行
        # 3. 强烈建议将每个任务的输出重定向到独立文件，避免混乱
        CUDA_VISIBLE_DEVICES=4 python run_train_new.py \
            --task tm \
            --train_path ../data/Tm/Tm_new_Train.csv \
            --test_path ../data/Tm/Tm_new_Test.csv \
            --seq_model "$model" \
            --lr 0.0001 \
            --cv $cv &
    done
done

# 等待所有剩余的后台任务执行完毕
echo "所有任务已启动，等待它们全部完成..."
wait

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))

echo "-----------------------------------"
echo "所有并行任务已执行完毕！"
echo "总耗时: $elapsed 秒"
echo "-----------------------------------"