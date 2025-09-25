######################################################################
# devide the kcat and km
# max replace value
import pandas as pd
import re
from tqdm import tqdm
input_file = 'kcat_km_values.csv'
output_file1 = 'kcat_data.csv'
output_file2 ='km_data.csv'
data = pd.read_csv(input_file)
data.drop(columns=['LigandStructureId', 'Literature'], inplace=True)
if 'Commentary' in data.columns:
    data.rename(columns={'Commentary': 'enzymeType'}, inplace=True)
# 初始化两个空的DataFrame，用于存储不同类型的数据
data_kcat = pd.DataFrame(columns=data.columns)
data_km = pd.DataFrame(columns=data.columns)
# 遍历数据行
for index, row in tqdm(data.iterrows()):
    if row['Value'] <= 0:
        continue  # 如果Value小于0，则跳过此行
    desc = str(row['enzymeType']).lower()
    if 'mutant' in desc or 'mutated' in desc:
        mutant = re.findall('[A-Z]\d+[A-Z]', desc)  # 查找所有变异
        if len(mutant) >= 1:  # 如果存在多个变异,divide
            enzymeType = '/'.join(mutant)
        else:
            continue
    else:
        enzymeType = 'wildtype'

    row['enzymeType'] = enzymeType  # 用enzymeType替换Commentary
    if pd.notnull(row['Maximum']) and row['Maximum'] != '':
        row['Value'] = row['Maximum']  # 如果Maximum存在，替换Value
    # 根据Type字段分配到不同的DataFrame
    if row['Type'].lower() == 'kcat':
        data_kcat = pd.concat([data_kcat, pd.DataFrame([row])], ignore_index=True)
    elif row['Type'].lower() == 'km':
        data_km = pd.concat([data_km, pd.DataFrame([row])], ignore_index=True)
data_kcat.drop(columns=['Maximum'], inplace=True)
data_km.drop(columns=['Maximum'], inplace=True)
data_kcat.to_csv(output_file1, index=False)
data_km.to_csv(output_file2, index=False)