import pandas as pd
data = pd.read_csv(f'tomer_0530.csv')
data_1 = pd.read_csv(f'Seq2Topt_topt_identity_new.csv')
data['identity'] = data_1['identity']
data.to_csv(f'tomer_0530.csv', index=False)