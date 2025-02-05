import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Conv2D, Flatten, Dropout, BatchNormalization, Add, Concatenate, MaxPooling1D, AveragePooling1D #, LocallyConnected2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.models import load_model

# Configure logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,  # Logs will be saved to this file
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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


def notch_filter(pressure, PMT, samp_freq=10000, notch_freq=60.0, quality_factor=20.0):
    """Apply a notch filter to remove noise from the signals."""
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, samp_freq)
    return signal.filtfilt(b_notch, a_notch, pressure), signal.filtfilt(b_notch, a_notch, PMT)


def get_data(cache_file, filenames, stability_label_dict, window_size, duration_sample_ms):
    """Load and preprocess data, or load from cache if available."""
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("There is not data available. Run the script called load_and_preprocess_transient.py to generate the data.")
    
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


def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach, "transient",f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Generate the cache file path
    cache_file = os.path.join(folder_path, "data.pkl")
    
    return cache_file


def get_log_file_path(project_root, window_size, approach):
    """Generate the log file path based on window size and approach."""
    # Create the logs folder if it doesn't exist
    logs_folder = os.path.join(project_root, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    
    # Generate the log file name
    log_file = os.path.join(logs_folder, f"{approach}_transient_window_{window_size}ms.log")
    
    return log_file


if __name__ == "__main__":
    args = parse_arguments()
    print(tf.__version__)
    print(keras.__version__)

    # Generate the log file path and set up logging
    log_file = get_log_file_path(args.project_root, args.window_size, args.approach)
    setup_logging(log_file)
    
    # Generate the cache file path
    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    
    # Load and preprocess data
    stability_label_dict=""
    transient_path=os.path.join(args.project_root,"data","raw","transient","hidrogen")
    filenames = [os.path.join(transient_path,"open_25kW_600slpm_70%to80%.csv"), os.path.join(transient_path,"open_25kW_600slpm_80%to70%.csv")]
    inputsALL, outputsALL_label = get_data(cache_file, filenames, stability_label_dict, args.window_size, args.duration_sample_ms)
    # # Log the shapes of the processed data
    logging.info(f"Processed data shapes - Inputs: {inputsALL.shape}, Outputs: {outputsALL_label.shape}")


    model_path=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms","model_2","lstm_model_ts.keras")
    model = load_model(model_path)
    # model = load_model(model_path, custom_objects={'InputLayer': tf.keras.layers.InputLayer})
    
    
    # Evaluate the model
    # loss, accuracy = model.evaluate(inputsALL, outputsALL_label, verbose=2)
    
    # Log the evaluation results
    # logging.info(f"Model evaluation - Loss: {loss}, Accuracy: {accuracy}")
    # print(f"Model evaluation - Loss: {loss}, Accuracy: {accuracy}")
    # model = tf.keras.models.load_model(model_path)
    # model = load_model(model_path, custom_objects={'InputLayer': tf.keras.layers.InputLayer})
    

    # with tf.keras.utils.custom_object_scope({'InputLayer': tf.keras.layers.InputLayer}):
    #     model = load_model(model_path)
    predictions = model.predict(inputsALL)
    predictions = (predictions > 0.5).astype(int) 

    accuracy = accuracy_score(outputsALL_label, predictions)
    
    # logging.info(f"Predictions: {predictions}")
    logging.info(f"Accuracy: {accuracy}")
    # print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(outputsALL_label, predictions)
    logging.info(f"Confusion Matrix:\n{cm}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Unstable'])
    disp.plot(cmap='Blues')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(outputsALL_label, label='Actual')
    plt.plot(predictions, label='Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Model Predictions vs Actual Values')
    plt.legend()
    plt.show()