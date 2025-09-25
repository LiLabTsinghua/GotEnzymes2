import json
import numpy as np
import matplotlib.pyplot as plt

# 读取JSON文件并解析数据
with open('../../Data/database/Kcat_combination_0732_wildtype_mutant.json', 'r') as f:
    data = json.load(f)


# 提取所需的字段
organism = [entry['Organism'] for entry in data]
ec_number = [entry['ECNumber'] for entry in data]
values = [entry['Value'] for entry in data]
type_ = [entry['Type'] for entry in data]
substrate = [entry['Substrate'] for entry in data]

# 整合数据
alldata = list(zip(organism, ec_number, values, type_, substrate))

# 获取唯一的物种名称
species = list(set([entry[0] for entry in alldata]))

# 初始化计数器
x = np.zeros((len(species), 3))

# 计算每个物种的wildtype和mutant类型的数量
for i, sp in enumerate(species):
    x[i, 0] = sum(1 for entry in alldata if entry[0] == sp and entry[3] == 'wildtype')
    x[i, 1] = sum(1 for entry in alldata if entry[0] == sp and entry[3] == 'mutant')
    x[i, 2] = sum(1 for entry in alldata if entry[0] == sp)

# 排序物种按wildtype数量降序排列
sorted_indices = np.argsort(x[:, 0])[::-1]
species = np.array(species)[sorted_indices]
x = x[sorted_indices, :]

# 绘制竖向堆叠柱状图
top_species = 20
data_to_plot = np.vstack([x[:top_species, :2], np.sum(x[top_species:, :2], axis=0)])
species_to_plot = list(species[:top_species]) + ['Others']

plt.barh(range(1, 22), data_to_plot[:, 0], height=0.5, label='Wildtype', linewidth=0.5)
plt.barh(range(1, 22), data_to_plot[:, 1], height=0.5, left=data_to_plot[:, 0], label='Mutant', linewidth=0.5)

plt.yticks(ticks=range(1, 22), labels=species_to_plot)
plt.xlabel(r'$\it{k}_{cat}$ number')
plt.legend(loc='upper right', fontsize=6)
plt.gcf().set_size_inches(5, 8)  # 调整图片尺寸
plt.subplots_adjust(left=0.5, right=0.95, top=0.95, bottom=0.1)  # 调整左侧边距
plt.grid(False)

plt.savefig("../../Results/figures/SuppleFig1c.pdf")
