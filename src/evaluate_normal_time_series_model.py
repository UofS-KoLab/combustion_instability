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
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("There is not data available. Run the script called load_and_preprocess_transient.py to generate the data.")
    
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

    model_path=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms",args.model_name,"lstm_model_ts.keras")
    model = load_model(model_path)

    # Evaluate the model
    predictions=[]
    cache_prediction_file=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms",args.model_name,"predictions.pkl")
    if os.path.exists(cache_prediction_file):
        logging.info("Loading data from cache...")
        predictions = load(cache_prediction_file)
    else:
        predictions = model.predict(inputsALL)
        predictions = (predictions > 0.5).astype(int) 
        dump((predictions), cache_prediction_file)
    
    accuracy = accuracy_score(outputsALL_label, predictions)
    
    logging.info(f"Accuracy: {accuracy}")
    print(f"Accuracy: {accuracy}")

    cm = confusion_matrix(outputsALL_label, predictions)
    logging.info(f"Confusion Matrix:\n{cm}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Unstable'])
    disp.plot(cmap='Blues')

    # Specify the folder and filename
    plot_path = os.path.join(args.project_root, "plot", args.approach)

    # Create the folder if it doesn't exist
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    # Save the plot
    plt.savefig(plot_path + f"/confusion_matrix_{args.window_size}ms_{args.model_name}.png", bbox_inches='tight', pad_inches=0.1)
    plt.show()
