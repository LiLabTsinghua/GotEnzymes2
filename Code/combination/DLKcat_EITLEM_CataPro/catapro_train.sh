#!/bin/bash

# 定义模型列表
ulimit -n 4096
smi_model_list=('maccskeys' 'smitrans' 'molgen' 'molebert' 'unimolv1' 'unimolv2' 'chemberta2' 'ecfp' 'rdkitfp')
seq_model_list=('esm2')
# seq_model_list=('esm2')
for smi_model in "${smi_model_list[@]}"
do
   for seq_model in "${seq_model_list[@]}"
   do
      python catapro_train.py -t catapro -m "MACCSKeys" -d 1 -smi "$smi_model" -seq "$seq_model" -cv 4
   done
done