#!/bin/bash

# 定义模型路径和参数文件
model_path="../data/performances/model_latentdim=40_outlayer=4_rmsetest=0.8854_rmsedev=0.908.pth"
param_dict_pkl="../data/hyparams/param_2.pkl"

# 循环不同的输入和输出配置
for i in {1..385}
do
    input_file="../data/input_${i}.csv"
    output_dir="../data/output_${i}"

    # 调用 predict.py 脚本
    python predict.py \
      --model_path $model_path \
      --param_dict_pkl $param_dict_pkl \
      --input $input_file \
      --output $output_dir \
      --has_label False

    echo "Completed processing for input $input_file"
done
