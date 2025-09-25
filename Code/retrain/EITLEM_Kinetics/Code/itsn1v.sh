#!/bin/bash

# 定义模型列表
ulimit -n 4096

cv_list=('0')

for cv in "${cv_list[@]}"
do
    python itsn_1v.py -i 1 -t eitlem -m "MACCSKeys" -d 4 -cv "$cv"
done
