import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging

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

    NUM_SEGMENTS_MIN = 120  # Minimum number of segments
    NUM_SEGMENTS_MAX = 120  # Maximum number of segments
    NUM_SEGMENTS_INC = 1    # Segment increment
    data = []
    outputsALLr = []

    SEGMENT_NUM= duration_sample_ms / window_size
    amount_of_samples=int(len(filenames)*SEGMENT_NUM)
    total_points=120000
    amount_points_in_segment=int(total_points/SEGMENT_NUM)
    logging.info(f"The inputs_all shape should be: {amount_of_samples}, {amount_points_in_segment}, 2")

    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        inputs_all = np.empty((12480, 3000, 2))  # Initialize array for input data
        outputs_all = []
        k = 0

        for i, file_path in enumerate(filenames):
            logging.info(f"Processing file #{i}: {file_path}")
            file_name = file_path.split('\\')[-1].replace(".csv", "")

            if file_name not in stability_label_dict:
                logging.info(f"Cannot find the filename in the label dictionary... Skipping {file_name}")
                continue
            label = stability_label_dict[file_name]


            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            
            expected_header = "time,p3,pmt"
            if first_line == expected_header:
                acoustic_data = pd.read_csv(file_path,names=['time', 'p3', 'pmt'], header=None, skiprows=1)
            else:    
                print("Not expected header")
            
            time = acoustic_data['time'].to_numpy()

            pressure, PMT = notch_filter(acoustic_data['p3'], acoustic_data['pmt'])

            for total_segments in range(NUM_SEGMENTS_MIN, NUM_SEGMENTS_MAX + 1, NUM_SEGMENTS_INC):
                time_segment_size = int(len(time) / total_segments)
                pressure_segment_size = int(len(pressure) / total_segments)
                PMT_segment_size = int(len(PMT) / total_segments)

                logging.info(f"time_segment_size: {time_segment_size}, {pressure_segment_size}, {PMT_segment_size}")

                for current_segment in range(1, total_segments + 1):
                    time_s = time[(current_segment - 1) * time_segment_size:current_segment * time_segment_size]
                    pressure_s = pressure[(current_segment - 1) * pressure_segment_size:current_segment * pressure_segment_size]
                    PMT_s = PMT[(current_segment - 1) * PMT_segment_size:current_segment * PMT_segment_size]

                    for j in range(time_segment_size):
                        inputs_all[k, j, 0] = pressure_s[j]
                        inputs_all[k, j, 1] = PMT_s[j]
                    k += 1

                    outputsALLr.append(label)
            logging.info(f"Processed {file_name} with {len(time)} time samples.")
        
        outputs_all = np.array(outputsALLr)
        dump((inputs_all, outputs_all), cache_file)
        data = (inputs_all, outputs_all)
        logging.info(f"Data saved to {cache_file}.")
    
    return data


def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach, f"window_{window_size}ms")
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
    stability_label_dict = load_stability_labels(args.stability_file)
    filenames = get_filenames(args.data_root)
    inputsALL, outputsALL_label = get_data(cache_file, filenames, stability_label_dict, args.window_size, args.duration_sample_ms)
    
    # # Log the shapes of the processed data
    logging.info(f"Processed data shapes - Inputs: {inputsALL.shape}, Outputs: {outputsALL_label.shape}")