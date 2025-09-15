#!/bin/bash

# 定义模型列表
ulimit -n 4096
smi_model_list=('maccskeys')
seq_model_list=('prott5')
# seq_model_list=('esm2')
for smi_model in "${smi_model_list[@]}"
do
   for seq_model in "${seq_model_list[@]}"
   do
      python catapro_train_org.py -t catapro -m "MACCSKeys" -d 1 -smi "$smi_model" -seq "$seq_model" -cv 4
   done
done