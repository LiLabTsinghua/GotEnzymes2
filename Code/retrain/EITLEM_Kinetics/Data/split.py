# 用来转移数据集中可能泄露的部分
import torch

kcat_all = list(torch.load("KCATTrainPairInfo"))
kcat_test = list(torch.load("KCATTestPairInfo"))
kcat_all.extend(kcat_test)
print(len(kcat_all))

km_all = list(torch.load("KMTrainPairInfo"))
km_test = list(torch.load("KMTestPairInfo"))
km_all.extend(km_test)
print(len(km_all))

kkm_all = list(torch.load("KKMTrainPairInfo"))
kkm_test = list(torch.load("KKMTestPairInfo"))
kkm_all.extend(kkm_test)

kcat_km_all = kcat_all + km_all
kcat_km_pair_list = [[x[0], x[1]] for x in kcat_km_all]
print(len(kcat_km_all))
kkm_test_number = int(len(kkm_all)*0.2)

#生成kkm数据集
kkm_new_test_list = [] # kkm测试集
for kkm in kkm_all:
    if len(kkm_new_test_list) < kkm_test_number:
        if [kkm[0], kkm[1]] in kcat_km_pair_list:
            kkm_new_test_list.append(kkm)
kkm_new_train_list = [x for x in kkm_all if x not in kkm_new_test_list] #kkm训练集
new_kkm_test_pair_list = [[x[0], x[1]] for x in kkm_new_test_list]

kkm_train_kcat = kcat_all + kkm_new_train_list
kkm_train_kcat_pair_list = [[x[0], x[1]] for x in kkm_train_kcat]

kkm_train_km= km_all + kkm_new_train_list
kkm_train_km_pair_list = [[x[0], x[1]] for x in kkm_train_km]

# 生成km数据集
km_test_number = int(len(km_all)*0.2)
km_new_test_list = []
for km in km_all:
    if [km[0], km[1]] in new_kkm_test_pair_list:
        km_new_test_list.append(km)
for km in km_all:
    if len(km_new_test_list) < km_test_number:
        if [km[0], km[1]] not in new_kkm_test_pair_list and [km[0], km[1]] not in kkm_train_kcat_pair_list:
            km_new_test_list.append(km)
km_new_train_list = [x for x in km_all if x not in km_new_test_list]

# 生成kcat数据集
kcat_test_number = int(len(kcat_all)*0.2)
kcat_new_test_list = []
for kcat in kcat_all:
    if [kcat[0], kcat[1]] in new_kkm_test_pair_list:
        kcat_new_test_list.append(kcat)
for kcat in kcat_all:
    if len(kcat_new_test_list) < kcat_test_number:
        if [kcat[0], kcat[1]] not in new_kkm_test_pair_list and [kcat[0], kcat[1]] not in kkm_train_km_pair_list:
            kcat_new_test_list.append(kcat)
kcat_new_train_list = [x for x in kcat_all if x not in kcat_new_test_list]

for i in kcat_new_train_list:
    if i not in kcat_all:
        print('出错')
        break

for i in kcat_new_test_list:
    if i not in kcat_all:
        print('出错')
        break

for i in km_new_train_list:
    if i not in km_all:
        print('出错')
        break

for i in km_new_test_list:
    if i not in km_all:
        print('出错')
        break

for i in kkm_new_train_list:
    if i not in kkm_all:
        print('出错')
        break

for i in kkm_new_test_list:
    if i not in kkm_all:
        print('出错')
        break

torch.save(kcat_new_train_list, 'NewKcatTrainList')
torch.save(kcat_new_test_list, 'NewKcatTestList')
torch.save(kkm_new_train_list, 'NewKKMTrainList')
torch.save(kkm_new_test_list, 'NewKKMTestList')
torch.save(km_new_train_list, 'NewKMTrainList')
torch.save(km_new_test_list, 'NewKMTestList')