#!/bin/bash

# 定义模型列表
ulimit -n 4096
smi_model_list=('maccskeys' 'smitrans' 'molgen' 'molebert' 'unimolv1' 'unimolv2' 'chemberta2' 'ecfp' 'rdkitfp')
seq_model_list=('esm2' 'esm1b' 'esm3b' 'esm15b' 'prott5' 'prollama')

for smi_model in "${smi_model_list[@]}"
do
   for seq_model in "${seq_model_list[@]}"
   do
        python train_.py -i 1 -t test1 -m "MACCSKeys" -d 1 -smi "$smi_model" -seq "$seq_model"
   done
done
# python train_.py -i 1 -t 250219 -m "MACCSKeys" -d 1 -smi "maccskeys" -seq "esm15b"