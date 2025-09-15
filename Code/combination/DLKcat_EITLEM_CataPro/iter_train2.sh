#!/bin/bash

# 定义模型列表
ulimit -n 4096

cv_list=('2')

for cv in "${cv_list[@]}"
do
    python iter_train_scripts_newduoka.py -i 1 -t eitlem -m "MACCSKeys" -cv "$cv"
done
# python train_.py -i 1 -t 250219 -m "MACCSKeys" -d 1 -smi "maccskeys" -seq "esm15b"