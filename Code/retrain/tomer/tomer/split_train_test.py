import pandas as pd
from sklearn.model_selection import train_test_split
data = pd.read_csv('data/sequence_ogt_topt.csv')
train_data, test_data = train_test_split(data, test_size=0.1, random_state=0)
train_data.to_csv('data/train_data_state0.csv', index=False)
test_data.to_csv('data/test_data_state0.csv', index=False)