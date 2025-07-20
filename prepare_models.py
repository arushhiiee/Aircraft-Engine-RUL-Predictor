import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, GRU, Dense, Dropout, MaxPooling1D
from sklearn.preprocessing import MinMaxScaler
import os
import logging
import joblib
import json

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Parameters ---
SEQUENCE_LENGTH = 50
RUL_CAP = 130
DATASETS = ['FD001', 'FD002', 'FD003', 'FD004']
MODEL_DIR = "models"

# --- Helper Functions (from original script) ---
def load_data(dataset_name):
    """Loads a single dataset from text files."""
    logging.info(f"Loading dataset: {dataset_name}...")
    cols = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f's{i}' for i in range(1, 22)]
    train_df = pd.read_csv(f'train_{dataset_name}.txt', sep='\s+', header=None, names=cols)
    return train_df

def preprocess_and_get_artifacts(train_df):
    """Preprocesses data and returns the scaler and feature columns for saving."""
    logging.info("Preprocessing data and extracting artifacts...")
    
    # Calculate and Clip RUL
    max_cycles = train_df.groupby('id')['cycle'].max().reset_index()
    max_cycles.columns = ['id', 'max_cycle']
    train_df = pd.merge(train_df, max_cycles, on='id', how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop('max_cycle', axis=1, inplace=True)
    train_df['RUL'] = train_df['RUL'].clip(upper=RUL_CAP)

    # Identify and drop constant/useless sensors
    std_devs = train_df.std()
    constant_cols = std_devs[std_devs == 0].index.tolist()
    logging.info(f"Dropping constant columns: {constant_cols}")
    train_df.drop(columns=constant_cols, inplace=True)

    # Scale features and save the scaler and feature list
    feature_cols = [col for col in train_df.columns if col.startswith('setting') or col.startswith('s')]
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    
    return train_df, scaler, feature_cols

def create_sequences(df, sequence_length, feature_cols):
    """Generates time-series sequences for training."""
    sequences, labels = [], []
    for engine_id in df['id'].unique():
        engine_df = df[df['id'] == engine_id]
        for i in range(len(engine_df) - sequence_length + 1):
            seq = engine_df[feature_cols].iloc[i:i + sequence_length].values
            label = engine_df['RUL'].iloc[i + sequence_length - 1]
            sequences.append(seq)
            labels.append(label)
    return np.array(sequences), np.array(labels)

def build_advanced_model(input_shape):
    """Builds the CNN-GRU hybrid model."""
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'),
        MaxPooling1D(pool_size=2),
        GRU(64, return_sequences=True),
        Dropout(0.3),
        GRU(32, return_sequences=False),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# --- Main Execution Block ---
if __name__ == "__main__":
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        logging.info(f"Created directory: {MODEL_DIR}")

    for dataset in DATASETS:
        logging.info(f"--- Processing and Training for {dataset} ---")
        
        # --- Data Handling ---
        train_df_raw = load_data(dataset)
        train_df, scaler, feature_cols = preprocess_and_get_artifacts(train_df_raw.copy())
        
        X_train, y_train = create_sequences(train_df, SEQUENCE_LENGTH, feature_cols)
        
        # --- Model Training ---
        logging.info(f"Building and training model for {dataset}...")
        model = build_advanced_model((SEQUENCE_LENGTH, len(feature_cols)))
        model.summary(print_fn=logging.info)
        
        # Using a simple fit without callbacks for preparation script
        # A fixed number of epochs is used for consistency. This can be tuned.
        model.fit(
            X_train, y_train,
            epochs=40,
            batch_size=128,
            verbose=1
        )
        
        # --- Saving Artifacts ---
        logging.info(f"Saving model and artifacts for {dataset}...")
        
        # Save the trained model
        model.save(os.path.join(MODEL_DIR, f'model_{dataset}.keras'))
        
        # Save the scaler
        joblib.dump(scaler, os.path.join(MODEL_DIR, f'scaler_{dataset}.joblib'))
        
        # Save the feature columns list
        with open(os.path.join(MODEL_DIR, f'features_{dataset}.json'), 'w') as f:
            json.dump(feature_cols, f)
            
        logging.info(f"Successfully saved all artifacts for {dataset}.")

    logging.info("\n--- Model and artifact preparation complete! ---")
