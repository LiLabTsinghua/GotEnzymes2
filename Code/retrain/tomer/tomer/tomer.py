"""
tomer: A Python package to predict the optimum catalytic temperature (Topt) of proteins
with machine learning
""" 




# Imports
#============#

import numpy as np
import pandas as pd
import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

import tomer.resreg as resreg




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




def retrieve_model():
    """Return a tuple, (tomer_rebagg, scaler), containing the TOMER rebagg object 
    (from resreg) and the sklearn standard scaler object."""
    
    # Get dataset
    csv_data = os.path.join(this_dir, 'data', 'sequence_ogt_topt.csv') 
    data = pd.read_csv(csv_data, index_col=0)    
    
    # Get features and label data
    aac = np.array([getAAC(seq) for seq in data['sequence']])
    ogt = data['ogt'].values.reshape((data.shape[0],1))
    X = np.append(aac, ogt, axis=1)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data['topt'].values
    
    # Fit TOMER with Rebagg ensemble to all 2,917 sequences
    base_reg = DecisionTreeRegressor(random_state=0)
    tomer_rebagg = resreg.Rebagg(m=100, s=600, base_reg=base_reg)
    relevance = resreg.sigmoid_relevance(y, cl=None, ch=72.2)
    tomer_rebagg.fit(X, y, relevance=relevance, relevance_threshold=0.5, 
               sample_method='random_oversample', size_method='variation', random_state=0)
    
    return (tomer_rebagg, scaler)




tomer_rebagg, scaler = retrieve_model()




# Predict Topt of new sequences
#================================#

def read_fasta(fasta):
    """
    Read the protein sequences in a fasta file
    
    Parameters
    -----------
    fasta: str
    	Filename of fasta file containing protein sequences
    
    Returns
    ----------
    (list_of_headers, list_of_sequences)
    	A tuple of corresponding lists of  protein descriptions 
    	and protein sequences in fasta_file
    	
    """
    with open(fasta, 'r') as fast:
        headers, sequences = [], []
        for line in fast:
            if line.startswith('>'):
                head = line.replace('>','').strip()
                headers.append(head)
                sequences.append('')
            else :
                seq = line.strip()
                if len(seq) > 0:
                    sequences[-1] += seq
    return (headers, sequences)




def read_ogts(ogt_file):
    """
    Read the optimal growth temperatures (OGT) of organisms from a text file.
    The first line/row of the text file must be a header, and the OGT data must be in the 
    last column.

    
    Returns
    ---------
    ogt_list : list of ogt values
    """
    with open(ogt_file, 'r') as file:
        lines = file.readlines()
    lines = lines[1:]  # Remove header row
    ogts = [line.strip().split()[-1] for line in lines if line.strip()]
    return ogts
    



def get_features(sequence, ogt):
    """Return a 21-element vector of features corresponding to the amino acid composition
    and OGT"""
    aac = getAAC(sequence)
    features = np.append(aac, [ogt]).reshape(1, 21)
    features = scaler.transform(features)
    return features




def pred_seq_topt(sequence, ogt):
    """Predict the optimal catalytic temperature (Topt) of a single protein sequence.
    
    Parameters
    -----------
    sequence : str
        Amino acid sequence of protein
    ogt : float
        Optimal growth temperature in degrees Celsius
    
    Returns
    ---------
    (y_pred, y_err) : tuple
        y_pred is the predicted optimal catalytic temperature (Topt), and y_err is the 
        standard error of the mean of the predictions of the 100 base learners in the 
        bagging ensemble.
    
    Examples
    ---------
    >>> sequence = '''MKKQVVEVLVEGGKATPGPPLGPAIGPLGLNVKQVVDKINEATKEFAGMQVPVKIIVDPV
    TKQFEIEVGVPPTSQLIKKELGLEKGSGEPKHNIVGNLTMEQVIKIAKMKRSQMLALTLKAAAKEVIGTALSMGVTVE
    GKDPRIVQREIDEGVYDELFEKAEKE'''
    >>> ogt = 95
    >>> y_pred, y_err = pred_seq_topt(sequence, ogt) 

    """
    sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
    X = get_features(sequence, ogt)
    y_val = tomer_rebagg.predict(X)
    y_err = tomer_rebagg.pred_std/np.sqrt(100) # Standard error of the mean for 100 base learners
    return y_val[0], y_err[0]




def pred_fasta_topt(fasta_file, ogt_file):
    """Predict the optimal catalytic temperature of protein sequences in a fasta file.
    
    Parameters
    -----------
    fasta_file : str
        Fasta file containing amino acid sequences of proteins
    ogt_file : str
        File containing OGTs of protein source organisms. The first line/row of the file 
        must be a header, and the OGT data must be in the last column.
    
    Returns
    ---------
    df  : A dataframe
        First column is the headers of the sequences in fasta_file, the second column is
        the predicted catalytic optimum temperatures, and the third column is the standard
        error of the mean of predictions of the 100 base learners in the bagging ensemble.
    
    Examples
    ----------
    >>> fasta_file = 'test/sequences.fasta'
    >>> ogt_file = 'test/ogts.txt'
    >>> df = pred_fasta_topt(fasta_file, ogt_file)
    """
        
    headers, sequences = read_fasta(fasta_file)
    ogts = read_ogts(ogt_file)
    assert len(sequences) == len(ogts), ("Unequal number of sequences and OGTs")
    X = np.asarray([np.squeeze(get_features(sequence, ogt)) \
                    for sequence, ogt in zip(sequences, ogts)])
    y_pred = tomer_rebagg.predict(X)
    y_err = tomer_rebagg.pred_std/np.sqrt(100)
    df = pd.DataFrame([headers, list(y_pred), list(y_err)]).transpose()
    df.columns = ['ID', 'Topt', 'Std err']
    return df


