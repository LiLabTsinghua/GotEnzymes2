import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import resreg

# Constants
AALIST = list('ACDEFGHIKLMNPQRSTVWY')
THIS_DIR, THIS_FILENAME = os.path.split(__file__)

def get_aac(sequence):
    """Calculate amino acid composition for a protein sequence."""
    sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
    return np.array([sequence.count(x) for x in AALIST]) / len(sequence)

def load_model_and_data():
    """Load the TOMER model, scaler, and dataset."""
    # Load dataset
    # csv_path = 'data/sequence_ogt_topt.csv'
    csv_path = '../../Seq2Topt/data/train_complete.csv'
    data = pd.read_csv(csv_path, index_col=0)
    
    # Prepare features and labels
    aac = np.array([get_aac(seq) for seq in data['sequence']])
    ogt = data['ogt'].values.reshape(-1, 1)
    X = np.hstack([aac, ogt])
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    y = data['topt'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=0
    )
    
    # Initialize and train model
    base_reg = DecisionTreeRegressor(random_state=0)
    relevance = resreg.sigmoid_relevance(y, cl=None, ch=72.2)
    tomer_rebagg = resreg.Rebagg(m=100, s=600, base_reg=base_reg)
    tomer_rebagg.fit(
        X, y, 
        relevance=relevance, 
        relevance_threshold=0.5,
        sample_method='random_oversample', 
        size_method='variation', 
        random_state=0
    )
    
    return tomer_rebagg, scaler, data, X_train, X_test, y_train, y_test, X, y

def predict_topt(sequence, ogt, model, scaler):
    """Predict optimal temperature for a single protein sequence."""
    try:
        sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
        aac = get_aac(sequence)
        features = np.append(aac, [ogt]).reshape(1, -1)
        features = scaler.transform(features)
        y_pred = model.predict(features)
        y_err = model.pred_std[0] / np.sqrt(100)  # SEM for 100 base learners
        return y_pred[0], y_err
    except Exception as e:
        print(f"Prediction error for sequence (OGT={ogt}): {str(e)}")
        return None, None

def evaluate_on_test_data(test_path, model, scaler):
    """Evaluate model performance on new test data."""
    test_data = pd.read_csv(test_path)
    results = {
        'uniprot_id': [],
        'sequence': [],
        'ogt': [],
        'true_topt': [],
        'pred_topt': [],
        'pred_err': []
    }
    
    for _, row in test_data.iterrows():
        try:
            pred, err = predict_topt(row['sequence'], row['ogt'], model, scaler)
            results['uniprot_id'].append(row['uniprot_id'])
            results['sequence'].append(row['sequence'])
            results['ogt'].append(row['ogt'])
            results['true_topt'].append(row['topt'])
            results['pred_topt'].append(pred)
            results['pred_err'].append(err)
        except Exception as e:
            print(f"Error processing {row['uniprot_id']}: {str(e)}")
            continue
    
    # Calculate R2 score
    valid_preds = [p for p in results['pred_topt'] if p is not None]
    valid_true = [t for p, t in zip(results['pred_topt'], results['true_topt']) if p is not None]
    
    if valid_preds:
        r2 = r2_score(valid_true, valid_preds)
        print(f"R2 Score: {r2:.4f}")
    else:
        print("No valid predictions to calculate R2 score")
    
    return pd.DataFrame(results)

# Main execution
if __name__ == "__main__":
    # Load model and data
    model, scaler, data, X_train, X_test, y_train, y_test, X_full, y_full = load_model_and_data()
    
    # Evaluate on new test data
    test_path = '../../Seq2Topt/data/test_complete.csv'
    # test_path = 'data/test_data_state0.csv'
    results_df = evaluate_on_test_data(test_path, model, scaler)
    
    # Save results
    results_df.to_csv('tomer_0530.csv', index=False)
    
    # Optional: Predict for full dataset
    # full_preds, full_errs = model.predict(X_full), model.pred_std / np.sqrt(100)
    # data['Predicted Topt'] = full_preds
    # data['Std err'] = full_errs
    # data.to_csv('tomer_full_results.csv', index=False)