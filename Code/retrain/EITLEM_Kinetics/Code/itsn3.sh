#!/bin/bash

# 定义模型列表
ulimit -n 4096

cv_list=('3')

for cv in "${cv_list[@]}"
do
    python itsn.py -i 1 -t eitlem -m "MACCSKeys" -d 0 -cv "$cv"
done
