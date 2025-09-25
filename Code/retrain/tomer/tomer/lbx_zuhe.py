import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import resreg as resreg

# Constants
AALIST = list('ACDEFGHIKLMNPQRSTVWY')
THIS_DIR, THIS_FILENAME = os.path.split(__file__)

def get_aac(sequence):
    """Calculate amino acid composition for a protein sequence."""
    sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
    return np.array([sequence.count(x) for x in AALIST]) / len(sequence)

def retrieve_model(seq_model, seed):
    """Return model components including PLM embeddings."""
    plm_path = f'../../data/plm/{seq_model}.pkl'
    
    with open(plm_path, 'rb') as f:
        plm_embeddings = pickle.load(f)
    
    csv_data = os.path.join(THIS_DIR, 'data', 'sequence_ogt_topt.csv') 
    data = pd.read_csv(csv_data, index_col=0)
    
    # Process embeddings
    embeddings_list = [np.array(plm_embeddings[seq]) for seq in data['sequence']]
    embeddings = np.array(embeddings_list)
    ogt = data['ogt'].values.reshape(-1, 1)
    X = np.hstack([embeddings, ogt])
    
    # Standardize data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = data['topt'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
    
    # Train model
    base_reg = DecisionTreeRegressor(random_state=0)
    relevance = resreg.sigmoid_relevance(y_train, cl=None, ch=72.2)
    tomer_rebagg = resreg.Rebagg(m=100, s=600, base_reg=base_reg)
    tomer_rebagg.fit(
        X_train, y_train, 
        relevance=relevance, 
        relevance_threshold=0.5,
        sample_method='random_oversample', 
        size_method='variation', 
        random_state=0
    )
    
    return tomer_rebagg, scaler, data, X_train, X_test, y_train, y_test, X, y

def predict_single_sequence(sequence, ogt, model, scaler, plm_embeddings):
    """Predict Topt for a single sequence with PLM embeddings."""
    try:
        # Clean sequence
        sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
        
        # Get PLM embedding
        embedding = np.array(plm_embeddings[sequence])
        
        # Create feature vector
        features = np.append(embedding, [ogt]).reshape(1, -1)
        features = scaler.transform(features)
        
        # Make prediction
        y_pred = model.predict(features)
        y_err = model.pred_std[0] / np.sqrt(100)  # Standard error
        
        return y_pred[0], y_err
    except Exception as e:
        print(f"Prediction error for sequence (OGT={ogt}): {str(e)}")
        return None, None

def process_dataset_sequentially(data, model, scaler, seq_model):
    """Process the entire dataset one sequence at a time."""
    # Load PLM embeddings
    plm_path = f'../../data/plm/{seq_model}.pkl'
    with open(plm_path, 'rb') as f:
        plm_embeddings = pickle.load(f)
    
    predictions = []
    errors = []
    
    for idx, row in data.iterrows():
        try:
            pred, err = predict_single_sequence(
                row['sequence'], 
                row['ogt'], 
                model, 
                scaler,
                plm_embeddings
            )
            predictions.append(pred)
            errors.append(err)
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            predictions.append(None)
            errors.append(None)
    
    return predictions, errors

if __name__ == '__main__':
    seq_model_list = ['prollama']  # Can be extended to other models
    
    for seq_model in seq_model_list:
        print(f"Processing model: {seq_model}")
        for seed in range(5):
            # Load model and data
            tomer_rebagg, scaler, data, X_train, X_test, y_train, y_test, X_full, y_full = retrieve_model(seq_model, seed)
            
            # Process dataset one sequence at a time
            y_pred_full, y_err_full = process_dataset_sequentially(data, tomer_rebagg, scaler, seq_model)
            
            # Add predictions to dataframe
            data['Predicted Topt'] = y_pred_full
            data['Std err'] = y_err_full
            
            # Mark test set
            train_indices, test_indices = train_test_split(data.index, test_size=0.1, random_state=seed)
            data['Test'] = 0
            data.loc[test_indices, 'Test'] = 1
            
            # Save results
            output_path = f'../results/tomer_{seq_model}_results_cv{seed}.csv'
            data.to_csv(output_path, index=False)
            print(f"Results saved to {output_path}")