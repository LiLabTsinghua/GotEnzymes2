#!/bin/bash
# 依次运行以下命令 all train test
# python gen_features.py --data "../data/EITLEM_KCAT_all.csv" --output ../data/all/ --has_dict False --has_label True
# python gen_features.py --data "../data/EITLEM_KCAT_train.csv" --output ../data/train/ --has_dict True --has_label True
# python gen_features.py --data ../data/EITLEM_KCAT_test.csv --output ../data/test/ --has_dict True --has_label True
python run_train_test.py --train_path ../data/train --test_path ../data/test --param_dict_pkl ../data/hyparams/param_2.pkl 