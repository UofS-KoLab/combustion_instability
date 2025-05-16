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
from tensorflow.keras.models import load_model
import time
from scipy.fft import fft
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
    parser.add_argument("--model_name", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    return parser.parse_args()

def get_filenames(data_root):
    """Get all Excel files in the specified directory."""
    return [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".csv")]

def get_data(cache_file):
    """Load and preprocess data, or load from cache if available."""
    data=[]
    print("carche",cache_file)
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("There is not data available. Run the script called load_and_preprocess_transient.py to generate the data.")
    print("here",data)
    inputsALL, outputsALL_label, output_name = data
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
    folder_path = os.path.join(project_root,"data", approach,"h2",f"window_{window_size}ms")
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
    log_file = os.path.join(logs_folder, f"{approach}_window_{window_size}ms.log")
    return log_file

def compute_fft(time_series, sampling_rate, max_freq=500):
    """Computes FFT and returns frequencies, magnitudes, powers, and phases."""
    n = len(time_series)
    
    fft_result = fft(time_series)
    freqs = np.fft.fftfreq(n, d=sampling_rate)

    positive_freqs = freqs[:n//2]
    positive_fft_result = fft_result[:n//2]

    mask = positive_freqs <= max_freq
    positive_freqs = positive_freqs[mask]
    positive_fft_result = positive_fft_result[mask]
    
    magnitude = np.abs(positive_fft_result)
    magnitude = np.round(magnitude, 5)
    power = magnitude ** 2
    phase = np.angle(positive_fft_result)

    return positive_freqs, magnitude, power, phase

if __name__ == "__main__":
    args = parse_arguments()

    # Generate the log file path and set up logging
    log_file = get_log_file_path(args.project_root, args.window_size, args.approach)
    setup_logging(log_file)
    
    # Generate the cache file path
    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    
    # Load and preprocess data
    stability_label_dict=""
    filenames = get_filenames(args.data_root)
    inputsALL, outputsALL_label = get_data(cache_file)
    # # Log the shapes of the processed data
    logging.info(f"Processed data shapes - Inputs: {inputsALL.shape}, Outputs: {outputsALL_label.shape}")

    # model_path=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms",args.model_name,"lstm_model_ts.keras")
    # model = load_model(model_path)

    print(inputsALL.shape)
    times_all=[]
    # for i in range(1000):
    #     input = inputsALL[i]
    #     start_time=time.time()
    #     input = input.reshape(1, input.shape[0], input.shape[1])
    #     prediction = model.predict(input)
    #     prediction = (prediction > 0.5).astype(int) 
    #     print(prediction)
    #     print(f"Time: {time.time()-start_time}")
    #     times_all.append(time.time()-start_time)

    # print(f"Mean time: {np.mean(times_all)}")
    # print(f"Max time: {np.max(times_all)}")
    # print(f"Min time: {np.min(times_all)}")
    model_path=os.path.join(args.project_root,"model","fft",f"window_{args.window_size}ms","model_20","lstm_model_fft.keras")
    model = load_model(model_path)
    for j in range(1000):
        pressure_s=inputsALL[j][:,0]
        PMT_s=inputsALL[j][:,1]
        start_time=time.time()
        freqs_pressure, magnitude_pressure, power_pressure, phase_pressure = compute_fft(pressure_s, 0.0001)
        freqs_pmt, magnitude_pmt, power_pmt, phase_pmt = compute_fft(PMT_s, 0.0001)
        
        mult_mag = magnitude_pressure*magnitude_pmt
        phase_subs = phase_pressure-phase_pmt

        combined_array = np.column_stack((mult_mag, phase_subs))[:16] #51
        
        combined_array = combined_array.reshape(1, 16, 2) #51
        
        # start_time=time.time()
        predictions = model.predict(combined_array)
        predictions = np.argmax(predictions, axis=1)
        times_all.append(time.time()-start_time)
        
    print(f"Mean time: {np.mean(times_all)}")
    print(f"Max time: {np.max(times_all)}")
    print(f"Min time: {np.min(times_all)}")
       

        

        