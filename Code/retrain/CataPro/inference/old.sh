#!/bin/bash

python predict.py \
    -inp_fpath ../samples/sample_inp.csv \
    -model_dpath ../models \
    -batch_size 64 \
    -device cuda:0 \
    -out_fpath catapro_test-pred.csv
