#!/bin/bash

python predict_new.py \
    -inp_fpath ../datasets/kcat-data_0.4simi-10fold.csv \
    -model_dpath ../models \
    -batch_size 64 \
    -device cuda:0 \
    -out_fpath kcat-pred.csv
