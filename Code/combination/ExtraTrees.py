import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
import os
import time
from joblib import dump

def load_input_from_pkl(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def Kcat_predict(data_df, smi_model, seq_model, kinetic_parameter):
    model_path = f'models/{smi_model}_{seq_model}_{kinetic_parameter}'
    os.makedirs(model_path, exist_ok=True)

    train_data = data_df[data_df['Test'] == 0]
    test_data = data_df[data_df['Test'] == 1]

    non_feature_cols = ['Sequence', 'Smiles', 'ECNumber', 'Organism', 'Substrate', 'type', 'Test']
    X_train = train_data.drop(columns=non_feature_cols)
    X_test = test_data.drop(columns=non_feature_cols)

    y_train = X_train.pop('Value')
    y_test = X_test.pop('Value')

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    start_time = time.time()
    model = ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test)
    test_r2 = r2_score(y_test, y_pred)

    # Save model
    dump(model, f"{model_path}/extratrees_model.joblib")

    print(f"\nTraining time: {end_time - start_time:.2f} seconds")
    print(f"Test R2 score: {test_r2:.4f}")

if __name__ == '__main__':
    smi_model_list = ['molgen']
    seq_model_list = ['prott5']
    kinetic_param_list = ['KCAT', 'KM', 'KKM']

    for seq_model in seq_model_list:
        sequence_embedding = load_input_from_pkl(f'../pretrain/saved_models/{seq_model}.pkl')
        for smi_model in smi_model_list:
            smiles_embedding = load_input_from_pkl(f'../pretrain/saved_models/{smi_model}.pkl')
            for kinetic_parameter in kinetic_param_list:
                print('Current combination:', smi_model, seq_model, kinetic_parameter)

                data = pd.read_csv(f'../data/EITLEM_{kinetic_parameter}.csv')
                print(f"Original data size: {len(data)}")

                # Filter invalid values and multi-part SMILES
                data['Value'] = np.log10(data['Value'].replace(0, 1e-12))
                filtered_data = data[
                    data['Value'].notna() &
                    (~data['Smiles'].str.contains('\.', na=False))
                ].copy()

                print(f"Filtered data size: {len(filtered_data)}")

                # Generate embeddings only for valid entries
                smiles_input = np.array([smiles_embedding[smiles] for smiles in filtered_data['Smiles']])
                sequence_input = np.array([sequence_embedding[seq] for seq in filtered_data['Sequence']])

                # Concatenate features
                feature_array = np.concatenate((smiles_input, sequence_input), axis=1)
                feature_df = pd.DataFrame(feature_array, index=filtered_data.index)
                feature_df.columns = [f'feature_{i}' for i in range(feature_df.shape[1])]

                # Combine with metadata and labels
                final_df = pd.concat([filtered_data.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)

                Kcat_predict(final_df, smi_model, seq_model, kinetic_parameter)