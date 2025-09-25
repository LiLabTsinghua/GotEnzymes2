# python seq2tm.py   --input ../data/Seq2Topt_PP_unique_input.csv --output Seq2Topt_PP_unique_input_tm &
# python seq2topt.py --input ../data/Seq2Topt_PP_unique_input.csv --output Seq2Topt_PP_unique_input_topt &
# python seq2tm.py   --input ../data/Seq2Topt_KMX_unique_input.csv --output Seq2Topt_KMX_unique_input_tm &
python seq2topt.py --input ../data/Tm/Tm_new_Test_cv0.csv --output test_cv0_tm_merged --cv 0 --type0 tm &
python seq2topt.py --input ../data/Tm/Tm_new_Test_cv1.csv --output test_cv1_tm_merged --cv 1 --type0 tm &
python seq2topt.py --input ../data/Tm/Tm_new_Test_cv2.csv --output test_cv2_tm_merged --cv 2 --type0 tm &
python seq2topt.py --input ../data/Tm/Tm_new_Test_cv3.csv --output test_cv3_tm_merged --cv 3 --type0 tm &
python seq2topt.py --input ../data/Tm/Tm_new_Test_cv4.csv --output test_cv4_tm_merged --cv 4 --type0 tm &