# Imports
#============#

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import resreg as resreg
import pickle
# Variables
#=================#

aalist = list('ACDEFGHIKLMNPQRSTVWY')
this_dir, this_filename = os.path.split(__file__)

# Get TOMER model
#======================#

def getAAC(seq):
    '''Calculate amino acid composition for a protein sequences (string)'''
    aac = np.array([seq.count(x) for x in aalist])/len(seq)
    return aac

def retrieve_model(seq_model):
    """Return a tuple, (tomer_rebagg, scaler, data), containing the TOMER rebagg object 
    (from resreg), the sklearn standard scaler object, and the full dataset."""
    
    plm_path = f'../../data/plm/{seq_model}.pkl'

    with open(plm_path, 'rb') as f:
        plm_embeddings = pickle.load(f)
    csv_data = os.path.join(this_dir, 'data', 'sequence_ogt_topt.csv') 
    data = pd.read_csv(csv_data, index_col=0)
    embeddings_list = [np.array(plm_embeddings[seq]) for seq in data['sequence']]
    embeddings = np.array(embeddings_list)
    # print('embd.shape',embeddings.shape)
    ogt = data['ogt'].values.reshape((data.shape[0], 1))
    # print('ogt.shape',ogt.shape) 
    # 将嵌入和 OGT 组合
    X = np.append(embeddings, ogt, axis=1)

    # 标准化数据
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)

    y = data['topt'].values
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    
    # Fit TOMER with Rebagg ensemble to the training set
    # base_reg = DecisionTreeRegressor(random_state=0)
    base_reg = ExtraTreesRegressor(n_jobs=80, random_state=0)
    tomer_rebagg = resreg.Rebagg(m=100, s=600, base_reg=base_reg)
    relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=72.2)
    tomer_rebagg.fit(X_train, y_train, relevance=relevance, relevance_threshold=0.5, 
               sample_method='random_oversample', size_method='variation', random_state=0)
    
    return (tomer_rebagg, scaler, data, X_train, X_test, y_train, y_test, X, y)


# Predict Topt of all sequences
#================================#

def predict_all_sequences(X_full):
    """Predict the optimal catalytic temperature (Topt) for the entire dataset.
    
    Parameters
    -----------
    X_full : numpy array
        Full dataset features
    
    Returns
    ---------
    (y_pred_full, y_err_full) : tuple
        y_pred_full is the predicted optimal catalytic temperature (Topt) for the entire dataset, 
        and y_err_full is the standard error of the mean of the predictions of the 100 base learners 
        in the bagging ensemble.
    """
    y_pred_full = tomer_rebagg.predict(X_full)
    y_err_full = tomer_rebagg.pred_std/np.sqrt(100) # Standard error of the mean for 100 base learners
    return y_pred_full, y_err_full

if __name__ == '__main__':
    seq_model_list = ['esm2', 'esm1b', 'esmc', 'esm1v', 'prott5']
    # seq_model_list = ['prott5']
    for seq_model in seq_model_list:
        tomer_rebagg, scaler, data, X_train, X_test, y_train, y_test, X_full, y_full = retrieve_model(seq_model)
        # Predict for the entire dataset
        y_pred_full, y_err_full = predict_all_sequences(X_full)

        # Add predictions to the original dataset
        data['Predicted Topt'] = y_pred_full
        data['Std err'] = y_err_full

        # Label training and test sets
        train_indices = data.index[data.index.isin(train_test_split(data.index, test_size=0.1, random_state=0)[0])]
        test_indices = data.index[data.index.isin(train_test_split(data.index, test_size=0.1, random_state=0)[1])]
        data['Test'] = 0
        data.loc[test_indices, 'Test'] = 1
        data.to_csv(f'../results/tomer_{seq_model}_results_etr.csv', index=False)

        