import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
r2l = []
for i in range(10):
    df = pd.read_csv(f'trainingceshikcat-pred_{i}.csv')
    r2 = r2_score(df['Label'], df.iloc[:,-3])
    r2l.append(r2)
    print(f"R2 score for fold {i}: {round(r2, 4)}")
print(f"Average R2 score: {round(sum(r2l)/len(r2l), 4)}")

r2l = []
for i in range(10):
    df = pd.read_csv(f'trainingceshikm-pred_{i}.csv')
    r2 = r2_score(df['Label'], df.iloc[:,-2])
    r2l.append(r2)
    print(f"R2 score for fold {i}: {round(r2, 4)}")
print(f"Average R2 score: {round(sum(r2l)/len(r2l), 4)}")

r2l = []
for i in range(10):
    df = pd.read_csv(f'kkm-pred_{i}.csv')
    r2 = r2_score(df['Label'], df.iloc[:,-1])
    r2l.append(r2)
    print(f"R2 score for fold {i}: {round(r2, 4)}")
print(f"Average R2 score: {round(sum(r2l)/len(r2l), 4)}")












'''
0.321
0.2663
0.2707
0.1336
0.2859
0.2447
0.176
0.2396
0.2236
0.2312'''
'''
0.4791
0.3555
0.3831
0.3942
0.4043
0.4013
0.4037
0.3972
0.3487
0.3768'''
'''
0.1271
0.1147
0.1867
0.1591
0.1911
0.1888
0.13
0.211
0.183
0.1214'''