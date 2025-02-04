import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Conv2D, Flatten, Dropout, BatchNormalization, Add, Concatenate, MaxPooling1D, AveragePooling1D #, LocallyConnected2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam, RMSprop #for windows
#from tensorflow.keras.optimizers import Adam, RMSprop #for linux
from tensorflow.keras import mixed_precision
import keras_tuner as kt
from tensorflow.keras.optimizers.legacy import Adam

def build_tunable_model(hp):
    # with tf.device('/GPU:0'):
        """Build a tunable model using Keras Tuner."""
        # Define hyperparameters to tune
        n_regularizer_k = hp.Float('n_regularizer_k', min_value=1e-5, max_value=1e-2, sampling='log')
        n_regularizer_r = hp.Float('n_regularizer_r', min_value=1e-5, max_value=1e-2, sampling='log')
        n_dropout = hp.Float('n_dropout', min_value=0.1, max_value=0.5, step=0.1)
        n_depth = hp.Int('n_depth', min_value=1, max_value=5, step=1)
        learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
        batch_size = hp.Choice('batch_size', values=[16, 32, 64])

        # Define the model
        cvt_in = tf.keras.Input(shape=(1000, 2), batch_size=batch_size)
        cvt_b = Bidirectional(LSTM(400, kernel_regularizer=l2(n_regularizer_k), recurrent_regularizer=l2(n_regularizer_r), 
                                return_sequences=True, stateful=True))(cvt_in)
        cvt_b = Dropout(n_dropout)(cvt_b)
        cvt_b = BatchNormalization()(cvt_b)

        for i in range(n_depth):
            cvt_b1 = Bidirectional(LSTM(400, kernel_regularizer=l2(n_regularizer_k), recurrent_regularizer=l2(n_regularizer_r), 
                                        return_sequences=True, stateful=True))(cvt_b)
            cvt_b1 = Dropout(n_dropout)(cvt_b1)
            cvt_plus = Add()([cvt_b, cvt_b1])
            cvt_b = BatchNormalization()(cvt_plus)

        cvt_f = Flatten()(cvt_b)
        cvt_out = Dense(24)(cvt_f)
        cvt_out = Activation('relu')(cvt_out)
        cvt_out = Dropout(n_dropout)(cvt_out)
        cvt_out = BatchNormalization()(cvt_out)
        cvt_out = Dense(1)(cvt_out)
        cvt_out = Activation('sigmoid')(cvt_out)

        model = tf.keras.Model(cvt_in, cvt_out)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[tf.keras.metrics.Recall()]
        )
        return model


def tune_hyperparameters(X_train, y_train, X_val, y_val):
    """Perform hyperparameter tuning using Keras Tuner."""
    # tuner = kt.RandomSearch(
    #     build_tunable_model,
    #     objective='val_recall',  # Maximize validation recall
    #     max_trials=20,  # Number of hyperparameter combinations to try
    #     executions_per_trial=2,  # Number of models to train per trial
    #     directory='hyperparameter_tuning',
    #     project_name='combustion_instability'
    # )
    tuner = kt.Hyperband(
        build_tunable_model,
        objective='val_recall',
        max_epochs=50,
        factor=3,
        directory='hyperparameter_tuning',
        project_name='combustion_instability',
        distribution_strategy=tf.distribute.MirroredStrategy(),
    )

    # Perform the search
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=32,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)]
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    print(f"n_regularizer_k: {best_hps.get('n_regularizer_k')}")
    print(f"n_regularizer_r: {best_hps.get('n_regularizer_r')}")
    print(f"n_dropout: {best_hps.get('n_dropout')}")
    print(f"n_depth: {best_hps.get('n_depth')}")
    print(f"learning_rate: {best_hps.get('learning_rate')}")
    print(f"batch_size: {best_hps.get('batch_size')}")

    return best_hps


#For windows
# Check if TensorFlow detects a GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU is available!")
    print("Details of GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected. TensorFlow is using the CPU.")
# Enable multi-threading
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

#For linux 
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.set_logical_device_configuration(
#                 gpu,
#                 [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 8)]  # Adjust memory limit if needed
#             )
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
#     except RuntimeError as e:
#         print(f"GPU Configuration Error: {e}")
# else:
#     print("No GPUs found, using CPU instead.")
    

# Configure logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,  # Logs will be saved to this file
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Disable unnecessary logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--stability_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--duration_sample_ms", type=int, required=True, help="Total duration of the sample of the time series in ms.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    return parser.parse_args()


def load_stability_labels(stability_filename):
    """Load stability labels from a CSV file."""
    stability_pd = pd.read_csv(stability_filename)
    logging.info(f"Found {len(stability_pd['Name'])} records in the labeling file.")
    
    stability_label_dict = {
        f"{row['Name']}": row['Stability']
        for _, row in stability_pd.iterrows()
    }
    return stability_label_dict


def get_filenames(data_root):
    """Get all Excel files in the specified directory."""
    return [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".csv")]

def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach, f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Generate the cache file path
    cache_file = os.path.join(folder_path, "data.pkl")

    print(os.path.join(folder_path, "data.pkl"))
    
    return cache_file

def get_data(cache_file, filenames, stability_label_dict):
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("No data to load.")
    
    inputsALL, outputsALL_label = data
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    label_mapping = {"Stable": 0, "Unstable": 1}
    outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)
    print(inputsALL.shape)
    print(outputsALL.shape)
    return inputsALL,outputsALL

def My_Model(time_steps, features, n_batch, n_regularizer_k, n_regularizer_r, n_dropout, n_depth, n_optimizer):
    # with tf.device('/GPU:0'):
        # Input layer
        cvt_in = tf.keras.Input(shape=(time_steps, features), batch_size=n_batch)
        
        # Bidirectional LSTM layers
        cvt_b = Bidirectional(LSTM(400, kernel_regularizer=l2(n_regularizer_k), recurrent_regularizer=l2(n_regularizer_r), 
                                return_sequences=True, stateful=True))(cvt_in)
        cvt_b = Dropout(n_dropout)(cvt_b)
        cvt_b = BatchNormalization()(cvt_b)
        
        # Additional LSTM layers
        for i in range(n_depth):
            cvt_b1 = Bidirectional(LSTM(400, kernel_regularizer=l2(n_regularizer_k), recurrent_regularizer=l2(n_regularizer_r), 
                                        return_sequences=True, stateful=True))(cvt_b)
            cvt_b1 = Dropout(n_dropout)(cvt_b1)
            cvt_plus = Add()([cvt_b, cvt_b1])
            cvt_b = BatchNormalization()(cvt_plus)
        
        # Flatten and dense layers
        cvt_f = Flatten()(cvt_b)
        cvt_out = Dense(24)(cvt_f)
        cvt_out = Activation('relu')(cvt_out)
        cvt_out = Dropout(n_dropout)(cvt_out)
        cvt_out = BatchNormalization()(cvt_out)
        
        # Output layer for binary classification
        cvt_out = Dense(1)(cvt_out)
        cvt_out = Activation('sigmoid')(cvt_out)
        
        # Create and compile the model
        model = Model(cvt_in, cvt_out)
        model.compile(loss='binary_crossentropy', optimizer=n_optimizer, metrics=[tf.keras.metrics.Recall()])
        
        return model

def create_incremented_folder(base_dir, base_name):
    count = 1
    new_folder = os.path.join(base_dir, f"{base_name}_{count}")
    while os.path.exists(new_folder):
        count += 1
        new_folder = os.path.join(base_dir, f"{base_name}_{count}")
    os.makedirs(new_folder)
    return new_folder


if __name__ == "__main__":
    args = parse_arguments()
    
    stability_label_dict = load_stability_labels(args.stability_file)
    filenames = get_filenames(args.data_root)

    cache_file=get_cache_file_path(args.project_root,args.window_size,args.approach)

    inputsALL, outputsALL = get_data(cache_file, filenames, stability_label_dict)

   # Manually split data for validation
    val_size = int(0.08 * len(inputsALL))
    val_size = val_size - (val_size % 32)  # Ensure divisible by batch size

    X_train = inputsALL[:-val_size]
    y_train = outputsALL[:-val_size]
    X_val = inputsALL[-val_size:]
    y_val = outputsALL[-val_size:]


    best_hps = tune_hyperparameters(X_train, y_train, X_val, y_val)

    # Log the best hyperparameters
    logging.info(f"Best hyperparameters: {best_hps}")
    print(best_hps)
