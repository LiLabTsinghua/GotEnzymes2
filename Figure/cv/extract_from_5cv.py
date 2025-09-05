import re
import json
from collections import defaultdict
# 文件列表
km = 'KKM'
file_names = [
    f'all_result_{km}_cv0.txt',
    f'all_result_{km}_cv1.txt',
    f'all_result_{km}_cv2.txt',
    f'all_result_{km}_cv3.txt',
    f'all_result_{km}_cv4.txt'
]
model_suffixes = ["esm1b", "esm1v", "esm2", "esmc", "prollama", "prott5"]

# 初始化最终结构
results = {}

# 正则表达式匹配csv文件名和对应的r2值
pattern = fr'\./(.*?)_{km}\.csv\s*Pcc:.*?r2:\s*(-?\d+\.\d+)'

for file_name in file_names:
    with open(file_name, 'r') as f:
        content = f.read()

    # 提取所有 csv 文件名和对应 r2 值
    matches = re.findall(pattern, content)

    for filename_part, r2_str in matches:
        # 分割出一级和二级模型名
        parts = filename_part.split('_')
        if len(parts) < 2:
            continue

        model1 = parts[0]
        model2 = parts[1]

        # 确保是有效的二级模型
        if model2 not in model_suffixes:
            continue

        r2_value = round(float(r2_str), 4)

        # 构建嵌套字典结构
        if model1 not in results:
            results[model1] = {}
        if model2 not in results[model1]:
            results[model1][model2] = []

        results[model1][model2].append(r2_value)
# 写入JSON文件
with open(f'{km}_unikp.json', 'w') as json_file:
    json.dump(results, json_file, indent=4)
