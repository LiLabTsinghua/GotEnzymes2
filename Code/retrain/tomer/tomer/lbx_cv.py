import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import resreg
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# Constants
AALIST = list('ACDEFGHIKLMNPQRSTVWY')
THIS_DIR, THIS_FILENAME = os.path.split(__file__)

def get_aac(sequence):
    """Calculate amino acid composition for a protein sequence."""
    sequence = sequence.replace(' ', '').replace('\n', '').replace('\t', '')
    return np.array([sequence.count(x) for x in AALIST]) / len(sequence)

def load_model_and_data(seed):
    """Load the TOMER model, scaler, and dataset."""
    # Load dataset
    csv_path = 'data/sequence_ogt_topt.csv'
    # csv_path = '../../Seq2Topt/data/train_complete.csv'
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
        X, y, test_size=0.1, random_state=seed
    )
    
    # Initialize and train model
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

def evaluate_on_test_data(test_data, model, scaler):
    """Evaluate model performance on new test data."""
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
    for seed in range(5):
        model, scaler, data, X_train, X_test, y_train, y_test, X_full, y_full = load_model_and_data(seed)
        
        # Evaluate on new test data
        test_path = '../../Seq2Topt/data/test_complete.csv'
        # test_path = 'data/test_data_state0.csv'
        # test_data = pd.read_csv(test_path)
        all_data = pd.read_csv('data/sequence_ogt_topt.csv')
        train_data, test_data = train_test_split(all_data, test_size=0.1, random_state=seed)
        results_df = evaluate_on_test_data(test_data, model, scaler)
        # Calculate additional metrics

        # Filter valid predictions
        valid_results = results_df.dropna(subset=['pred_topt', 'true_topt'])
        true_topt = valid_results['true_topt'].values
        pred_topt = valid_results['pred_topt'].values

        # Calculate metrics
        r2 = r2_score(true_topt, pred_topt)
        rmse = root_mean_squared_error(true_topt, pred_topt)
        mae = mean_absolute_error(true_topt, pred_topt)
        pcc, _ = pearsonr(true_topt, pred_topt)

        print(f"Seed {seed}:")
        print(f"R2: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"PCC: {pcc:.4f}")
    # Save results
    # results_df.to_csv('tomer_test_results.csv', index=False)
    
    # Optional: Predict for full dataset
    # full_preds, full_errs = model.predict(X_full), model.pred_std / np.sqrt(100)
    # data['Predicted Topt'] = full_preds
    # data['Std err'] = full_errs
    # data.to_csv('tomer_full_results.csv', index=False)