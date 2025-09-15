#!/bin/bash

# 定义模型列表
ulimit -n 4096
smi_model_list=('maccskeys' 'smitrans' 'molgen' 'molebert' 'unimolv1' 'unimolv2' 'chemberta2' 'ecfp' 'rdkitfp')
seq_model_list=('esm1v')
# seq_model_list=('esm2')
for smi_model in "${smi_model_list[@]}"
do
   for seq_model in "${seq_model_list[@]}"
   do
      python dlkcat_train.py -t dlkcat -m "MACCSKeys" -d 6 -smi "$smi_model" -seq "$seq_model"
   done
done
'esm1v': 1280