import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt
import time
import keras_tuner as kt

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Conv2D, Flatten, Dropout, BatchNormalization, Add, Concatenate, MaxPooling1D, AveragePooling1D #, LocallyConnected2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam, RMSprop #for windows
#from tensorflow.keras.optimizers import Adam, RMSprop #for linux
from tensorflow.keras import mixed_precision

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

# Disable unnecessary logging for windows
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
    # parser.add_argument('--lstm_units1', type=int, default=160, help="Number of units in the first LSTM layer.")
    # parser.add_argument('--lstm_units2', type=int, default=192, help="Number of units in the second LSTM layer.")
    # parser.add_argument('--dropout_rate1', type=float, default=0.3, help="Dropout rate for the first LSTM layer.")
    # parser.add_argument('--dropout_rate2', type=float, default=0.0, help="Dropout rate for the second LSTM layer.")
    # parser.add_argument('--l2_regularizer1', type=float, default=0.0099, help="L2 regularization for the first LSTM layer.")
    # parser.add_argument('--l2_regularizer2', type=float, default=0.0004, help="L2 regularization for the second LSTM layer.")
    # parser.add_argument('--learning_rate', type=float, default=0.008, help="Learning rate for the optimizer.")
    # parser.add_argument('--epochs', type=int, default=150, help="Number of epochs to train the model.")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split ratio.")
    # parser.add_argument('--patience', type=int, default=3, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--features', type=str, default="all", help='Comma-separated list of feature indices to select, or "all" to use all features.')
    
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
    folder_path = os.path.join(project_root,"data",args.approach, "h2", f"window_{window_size}ms")
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
    
    inputsALL, outputsALL_label, output_name = data
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    # label_mapping = {"Stable": 0, "Unstable": 1}
    # outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)
    

    df_new_cluster = pd.read_csv("C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\labels\\h2kmeans_thr_0.7_label.csv")
    new_outputSAll=np.zeros_like(outputsALL_label)
    
    for i in range(len(output_name)):
        # new_outputSAll[i]=df_new_cluster[df_new_cluster['filename']==output_name[i]]['cluster_label'].values[0]
        new_outputSAll[i]=df_new_cluster[df_new_cluster['filename']==output_name[i]]['label_stephany'].values[0]
    
    # new_outputSAll = new_outputSAll.astype(int)
    label_mapping = {'Stable': 0, 'Unstable': 1}
    # new_outputSAll = new_outputSAll.astype(int)
    # print(new_outputSAll)
    new_outputSAll = np.where(new_outputSAll == "Stable", 0, 
                 np.where(new_outputSAll == "Unstable", 1, new_outputSAll))

    # Convert the array to integers (if needed)
    new_outputSAll = new_outputSAll.astype(int)

    # Print the updated array to verify



    # new_outputSAll = new_outputSAll.tolist()
    # print(new_outputSAll)
    # outputsALL = np.array([label_mapping[label] for label in new_outputSAll], dtype=np.int32)

    


  

    # new_outputSAll[new_outputSAll == 1] = 4
    # # new_outputSAll[new_outputSAll == 0] = 1
    # new_outputSAll[new_outputSAll == 2] = 1 #0
    # new_outputSAll[new_outputSAll == 4] = 2

    outputsALL=np.array(new_outputSAll)
    print(inputsALL.shape)
    print(outputsALL.shape)


    return inputsALL,outputsALL, output_name

def compute_class_weights(y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    # class_weights_dict = {0: 0.1, 1: 0.9}
    return class_weights_dict

def build_model(hp):
    input_shape = (X_train.shape[1], X_train.shape[2])
    output_shape = len(np.unique(y_train))
    
    # model = Sequential([
    #     LSTM(hp.Int('lstm_units1', min_value=64, max_value=256, step=32),
    #          input_shape=input_shape, return_sequences=True, 
    #          kernel_regularizer=l2(hp.Float('l2_regularizer1', 1e-4, 1e-2, sampling='LOG'))),
    #     Dropout(hp.Float('dropout_rate1', 0.0, 0.5, step=0.1)),
        
    #     LSTM(hp.Int('lstm_units2', min_value=64, max_value=256, step=32),
    #          kernel_regularizer=l2(hp.Float('l2_regularizer2', 1e-4, 1e-2, sampling='LOG'))),
    #     Dropout(hp.Float('dropout_rate2', 0.0, 0.5, step=0.1)),
        
    #     Dense(output_shape, activation='softmax')
    # ])
    model = Sequential([
        LSTM(160,
             input_shape=input_shape, return_sequences=True, 
             kernel_regularizer=l2(hp.Float('l2_regularizer1', 1e-4, 1e-2, sampling='LOG'))),
         Dropout(hp.Float('dropout_rate1', 0.0, 0.5, step=0.1)),#Dropout(0.3),
        
        LSTM(192,
             kernel_regularizer=l2(hp.Float('l2_regularizer2', 1e-4, 1e-2, sampling='LOG'))),
        Dropout(hp.Float('dropout_rate2', 0.0, 0.5, step=0.1)),#Dropout(0.0),
        
        Dense(output_shape, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def save_model(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'lstm_model_fft.keras'))

def save_training_params(params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, 'training_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

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

    inputsALL, outputsALL , output_name= get_data(cache_file, filenames, stability_label_dict)

    # Ensure reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputsALL, outputsALL, test_size=args.test_size, random_state=args.seed)
    
    selected_features = list(map(int, args.features.split(',')))
    
    X_train = X_train[:, :, selected_features]
    X_test = X_test[:, :, selected_features]

    input_shape = X_train.shape[1:]  # Shape of each sample (timesteps, features)
    output_shape = len(np.unique(outputsALL))  # Assuming binary classification (stable or unstable)

    # Compute class weights
    class_weights_dict = compute_class_weights(y_train)
    print(class_weights_dict)

    # Initialize tuner
    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=50,
        factor=3,
        directory='tuning_results_label_stepahany_30ms', #100_3labels_gmm
        project_name='lstm_hyperparameter_tuning'
    )

    # Compute class weights
    class_weights = compute_class_weights(y_train)

    start_time = time.time()
    # Perform search
    tuner.search(X_train, y_train, epochs=30, validation_data=(X_test, y_test), 
                class_weight=class_weights, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    for hp in best_hps.values:
        print(f"{hp}: {best_hps.get(hp)}")

    Stop_time = time.time()
    print(f"Time taken: {Stop_time - start_time}")
    print(best_hps)