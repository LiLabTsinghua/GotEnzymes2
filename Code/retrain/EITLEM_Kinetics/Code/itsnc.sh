#!/bin/bash

# 定义模型列表
ulimit -n 4096

cv_list=('0')

for cv in "${cv_list[@]}"
do
    python itsn_c.py -i 1 -t eitlem -m "MACCSKeys" -d 3 -cv "$cv"
done
