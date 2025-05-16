import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import skew, kurtosis

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--old_data_file", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--stability_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument("--fuel_type", type=str, required=True, help="Type of fuel used in the experiment.")
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

def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach,args.fuel_type, f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Generate the cache file path
    cache_file = os.path.join(folder_path, "data.pkl")

    # print(os.path.join(folder_path, "data.pkl"))
    
    return cache_file

def get_data(cache_file):
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("No data to load.")
    
    inputsALL, outputsALL_label, output_name = data
    # inputsALL, outputsALL_label = data
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    label_mapping = {"Stable": 0, "Unstable": 1}
    outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)

    return inputsALL,outputsALL, output_name
    # return inputsALL,outputsALL

def get_old_data(old_data_file_path):
    # Split the columns argument into a list
    columns_to_pick = ['filename','case','heat_input','flow_rate','hydrogen_ratio','stability','rms','rms_pmt']
    # Construct the path to the CSV file
    csv_file_path = old_data_file_path
    # Read the CSV file and pick only the specified columns
    df = pd.read_csv(csv_file_path, usecols=columns_to_pick)
    df['filename'] = df['filename'].str.split('.csv').str[0]
    # Print the resulting DataFrame
    # print(df)
    return df

def calculate_rate_of_change(pressure_signal, time_interval):
    """
    Calculate the rate of change of a pressure signal.

    :param pressure_signal: List of pressure values.
    :param time_interval: Constant time interval between consecutive pressure readings.
    :return: List of rates of change.
    """
    rate_of_change = []
    
    for i in range(1, len(pressure_signal)):
        # Calculate the difference between consecutive pressure values
        delta_pressure = pressure_signal[i] - pressure_signal[i - 1]
        
        # Calculate the rate of change (delta_pressure / delta_time)
        rate = delta_pressure / time_interval
        
        # Append the rate of change to the list
        rate_of_change.append(rate)
    
    return rate_of_change

def extract_statistics(sample):
    """
    Extract various statistics from the given sample.
    """
    mean_sample = np.mean(sample)
    std_sample = np.std(sample)
    variance_sample = np.var(sample)
    skew_sample = skew(sample)
    kurtosis_sample = kurtosis(sample)
    max_sample = np.max(sample)
    min_sample = np.min(sample)
    peak_to_peak_sample = max_sample - min_sample
    rate_of_change_sample = np.mean(calculate_rate_of_change(sample, 0.0001))
    rms_sample = np.sqrt(np.mean(np.square(sample)))

    return {
        "mean": mean_sample,
        "std": std_sample,
        "variance": variance_sample,
        "skew": skew_sample,
        "kurtosis": kurtosis_sample,
        "max": max_sample,
        "min": min_sample,
        "peak_to_peak": peak_to_peak_sample,
        "rate_of_change": rate_of_change_sample,
        "rms": rms_sample
    }

if __name__ == "__main__":
    args = parse_arguments()
    # old_data=get_old_data(args.old_data_file)

    stability_label_dict = load_stability_labels(args.stability_file)
    names_files = list(stability_label_dict.keys())
    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    inputsALL, outputsALL , output_name= get_data(cache_file)
    # inputsALL, outputsALL = get_data(cache_file)
    print(inputsALL.shape)
    print(outputsALL.shape)

    rows=[]

    for i in range(inputsALL.shape[0]):
    # for i in range(1):
        sample = inputsALL[i]
        # print(type(sample), type(sample[:,0]))
        stats_pressure = extract_statistics(sample[:,0])
        stats_pmt= extract_statistics(sample[:,1])
        
                # Prepare the row to add to the DataFrame
        row = {
            'filename': output_name[i],
            'stability': stability_label_dict[output_name[i]],
            # 'filename': outputsALL[i],
            # 'stability': outputsALL[i],
            'mean_pressure': stats_pressure['mean'],
            'mean_pmt': stats_pmt['mean'],
            'std_pressure': stats_pressure['std'],
            'std_pmt': stats_pmt['std'],
            'variance_pressure': stats_pressure['variance'],
            'variance_pmt': stats_pmt['variance'],
            'skew_pressure': stats_pressure['skew'],
            'skew_pmt': stats_pmt['skew'],
            'kurtosis_pressure': stats_pressure['kurtosis'],
            'kurtosis_pmt': stats_pmt['kurtosis'],
            'max_pressure': stats_pressure['max'],
            'max_pmt': stats_pmt['max'],
            'min_pressure': stats_pressure['min'],
            'min_pmt': stats_pmt['min'],
            'peak_to_peak_pressure': stats_pressure['peak_to_peak'],
            'peak_to_peak_pmt': stats_pmt['peak_to_peak'],
            'rate_of_change_pressure': stats_pressure['rate_of_change'],
            'rate_of_change_pmt': stats_pmt['rate_of_change'],
            'rms_pressure': stats_pressure['rms'],
            'rms_pmt': stats_pmt['rms']
        }

        rows.append(row)

    df=pd.DataFrame(rows)
    print(df)
    df.to_csv(os.path.join(args.project_root,"data","stats",f"{args.fuel_type}_stats.csv"), index=False)

        



    # Plot inputsALL with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))

    # Plot the first dataset
    axs[0].plot(inputsALL[0, :, 0])
    axs[0].set_title('Dataset 1')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Value')

    # Plot the second dataset
    axs[1].plot(inputsALL[0, :, 1])
    axs[1].set_title('Dataset 2')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Value')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()