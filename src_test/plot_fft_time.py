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
    parser.add_argument("--fft_path", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--stability_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--cluster_label_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--fft_stats_info_file", type=str, required=True, help="Path to the stability labeling CSV file.")
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
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    label_mapping = {"Stable": 0, "Unstable": 1}
    outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)

    return inputsALL,outputsALL, output_name

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

    stability_label_dict = load_stability_labels(args.stability_file)
    names_files = list(stability_label_dict.keys())
    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    # cache_file_fft = get_cache_file_path(args.project_root, args.window_size, "fft")
    cache_file_fft = get_cache_file_path(args.project_root,500, "fft")
    inputsALL, outputsALL , output_name= get_data(cache_file)
    inputsALL_fft, outputsALL_fft, output_name_fft = get_data(cache_file_fft)
    print(inputsALL.shape)
    print(outputsALL.shape)

    stats_info_df = pd.read_csv(args.fft_stats_info_file)
    df = pd.read_csv(args.cluster_label_file)


    merged_df = pd.merge(df, stats_info_df, on='filename')

    # merged_df = merged_df[['filename', 'stability_x', 'cluster_label', 'skew_pmt',
    #                        'max_pressure','skew_pressure','peak_to_peak_pressure',
    #                        'std_pressure','kurtosis_pmt','min_pressure','max_pmt',
    #                         'std_pmt','peak_to_peak_pmt'
    #                        ]]
    # merged_df = merged_df[['filename', 'stability_x', 'cluster_label', 'sync_total_peaks',
    #                        'peak_freq_5_100_pmt','peak_mag_300_400_p','peak_mag_200_300_p',
    #                         'peak_freq_200_300_pmt','total_peaks_pressure','peak_mag_400_500_p',
    #                         'peak_freq_200_300_p','sync_detected','peak2_freq_pressure',
    #                        'peak1_freq_pressure','peak_freq_300_400_p'

    #                        ]]

    print(merged_df.head())

    


    # output_path_stats= os.path.join(args.project_root, "data", "stats", f"{args.fuel_type}_filter_cluster_fft_2.csv")
    # merged_df.to_csv(output_path_stats, index=False)
    # print(f"Merged DataFrame saved to {output_path_stats}")


    # Filter the DataFrame
    # filtered_df = df[(df['stability'] == 'Stable') & (df['cluster_label'] == 1)]

    # Get the list of filenames
    # filenames = filtered_df['filename'].tolist()
    filenames = df['filename'].tolist()

    
    print(len(filenames))
    print(filenames)

    for i in range(len(filenames)):
        # if filenames[i] =="open_20kW_400slpm_20_":
        # if filenames[i] ==filenames[0]:
        if filenames[i] =="open_30kW_400slpm_40_":
            index_sample = list(output_name).index(filenames[i])
            pressure_signal = inputsALL[index_sample, :, 0]
            max_pressure_index = np.argmax(pressure_signal)
            max_pressure_time = max_pressure_index  # Assuming time is the index


            index_sample=list(output_name).index(filenames[i])
            print(index_sample)
            # Plot inputsALL with subplots
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))

            # x = np.linspace(0, 1, inputsALL.shape[1])
            x = np.linspace(0, 1, 10000)
          
            axs[0].plot(0.23, 0.2, 'ro', )
            axs[0].plot(0.46, 0.2, 'ro', )
            axs[0].plot(0.69, 0.2, 'ro', )
            axs[0].plot(0.92, 0.2, 'ro', )

            axs[0].axvline(x=0.23, color='red', linestyle='--', linewidth=0.5)
            axs[0].axvline(x=0.46, color='red', linestyle='--', linewidth=0.5)
            axs[0].axvline(x=0.69, color='red', linestyle='--', linewidth=0.5)
            axs[0].axvline(x=0.92, color='red', linestyle='--', linewidth=0.5)
       
            axs[0].plot(x,inputsALL[index_sample, :10000, 0])
            axs[0].set_title('Pressure Signal')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Value')
            axs[0].legend()
            axs[0].set_xlim(0, 1)
            for xc in np.arange(0, 1.1, 0.1):
                axs[0].axvline(x=xc, color='gray', linestyle='--', linewidth=0.5)
           

         # Plot the first dataset
            for k in np.arange(0.148, 1.1, 0.1):
                print(k)
                axs[1].plot(k, 0.2, 'ro' )
                axs[1].axvline(x=k, color='red', linestyle='--', linewidth=0.5)
            # axs[1].plot(0.248, 0.2, 'ro', )
            # axs[1].plot(0.348, 0.2, 'ro', )
            # axs[1].plot(0.448, 0.2, 'ro', )

            # axs[1].axvline(x=0.23, color='red', linestyle='--', linewidth=0.5)
            # axs[1].axvline(x=0.46, color='red', linestyle='--', linewidth=0.5)
            # axs[1].axvline(x=0.69, color='red', linestyle='--', linewidth=0.5)
            # axs[1].axvline(x=0.92, color='red', linestyle='--', linewidth=0.5)
            
            # axs[0].annotate("predict", (0.23, 0.2), textcoords="offset points", xytext=(0,10), ha='center')
           
            
            axs[1].plot(x,inputsALL[index_sample, :10000, 0])
            axs[1].set_title('Pressure Signal')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Value')
            axs[1].legend()
            axs[1].set_xlim(0, 1)
            for xc in np.arange(0, 1.1, 0.1):
                axs[1].axvline(x=xc, color='gray', linestyle='--', linewidth=0.5)
            # axs[0].axvspan(0, 0.1, color='red', alpha=0.1)
 
            





        if filenames[i] =="open_30kW_400slpm_40_":           
                
            max_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'max_pressure'].values[0]
            peak_to_peak_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak_to_peak_pressure'].values[0]
            std_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'std_pressure'].values[0]
            skew_pmt=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'skew_pmt'].values[0]
            index_sample = list(output_name).index(filenames[i])
            print(index_sample)

            # Get the pressure signal and find the index of the maximum value
            pressure_signal = inputsALL[index_sample, :, 0]
            max_pressure_index = np.argmax(pressure_signal)
            max_pressure_time = max_pressure_index  # Assuming time is the index


            index_sample=list(output_name).index(filenames[i])
            print(index_sample)
            # Plot inputsALL with subplots
            fig, axs = plt.subplots(2, 1, figsize=(12, 8))

            x = np.linspace(0, 12, inputsALL.shape[1])
            # Plot the first dataset
            axs[0].plot(x,inputsALL[index_sample, :, 0], label=f'Max Pressure: {max_pressure_value:.2f}\n Peak to Peak Pressure: {peak_to_peak_pressure_value:.2f}\n Std Pressure: {std_pressure_value:.2f}\n Skew PMT: {skew_pmt:.2f}')
            axs[0].plot(x[max_pressure_time], pressure_signal[max_pressure_index], 'ro')  # Plot the max pressure point
            axs[0].set_title('Pressure Signal')
            axs[0].set_xlabel('Time')
            axs[0].set_ylabel('Value')
            axs[0].legend()
            axs[0].set_xlim(0, 12)

            max_pmt_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'max_pmt'].values[0]
            peak_to_peak_pmt_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak_to_peak_pmt'].values[0]
            std_pmt_value= stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'std_pmt'].values[0]

            pmt_signal = inputsALL[index_sample, :, 1]
            max_pmt_index = np.argmax(pmt_signal)
            max_pmt_time = max_pmt_index  # Assuming time is the index

            # Plot the second dataset
            axs[1].plot(x,inputsALL[index_sample, :, 1], label=f'Max PMT: {max_pmt_value:.2f}\n Peak to Peak PMT: {peak_to_peak_pmt_value:.2f}\n Std PMT: {std_pmt_value:.2f}')
            axs[1].plot(x[max_pmt_time], pmt_signal[max_pmt_index], 'ro')  # Plot the max pressure point
            axs[1].set_title('PMT Signal')
            axs[1].set_xlabel('Time')
            axs[1].set_ylabel('Value')
            axs[1].legend()
            axs[1].set_xlim(0, 12)

            plt.suptitle(f"Time Domain Features {filenames[i]}\n Stability: {stability_label_dict[filenames[i]]}\n New Cluster Label: {df['cluster_label'].iloc[i]}")
            # Adjust layout to prevent overlap
            plt.tight_layout()

            # Show the plot
            plt.show()
        
      
            print(inputsALL_fft.shape)
            print(outputsALL_fft.shape)
            print(output_name_fft.shape)
            index_sample = list(output_name_fft).index(filenames[i])
            x = np.linspace(0, 500, inputsALL_fft.shape[1])

            peak1_freq_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak1_freq_pressure'].values[0]
            peak2_freq_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak2_freq_pressure'].values[0]
            peak3_freq_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'freq_3_pressure'].values[0]

            sync_detected_range_1_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'sync_detected'].values[0]
            peak1_mag_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak1_mag_pressure'].values[0]
            peak2_mag_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak2_mag_pressure'].values[0]
            peak3_mag_pressure_value = stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'mag_3_pressure'].values[0]
            
            # peak1_freq_index = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak1_freq_pressure_value, atol=1e-5))[0][0]
            # peak2_freq_index = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak2_freq_pressure_value, atol=1e-5))[0][0]  
            # peak3_freq_index = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak3_freq_pressure_value, atol=1e-5))[0][0]  
            

            fig,axs=plt.subplots(2,1,figsize=(12,8))
            axs[0].plot(x,inputsALL_fft[index_sample, :, 1], label=f'Syncz: {sync_detected_range_1_value}\nPeak1 f:{peak1_freq_pressure_value:.2f} m:{peak1_mag_pressure_value:.2f}\n Peak2 f:{peak2_freq_pressure_value:.2f},m:{peak2_mag_pressure_value:.2f} \n Peak3 f:{peak3_freq_pressure_value:.2f},m:{peak3_mag_pressure_value:.2f}')
            # axs[0].plot(x[peak1_freq_index],inputsALL_fft[index_sample, peak1_freq_index, 1], 'ro')  # Plot the max pressure point
            # axs[0].plot(x[peak2_freq_index],inputsALL_fft[index_sample, peak2_freq_index, 1], 'bx')  # Plot the max pressure point 
            # axs[0].plot(x[peak3_freq_index],inputsALL_fft[index_sample, peak3_freq_index, 1], 'bx')  # Plot the max pressure point 
            
            axs[0].set_title('FFT Pressure Signal')
            axs[0].set_xlabel('Frequency')
            axs[0].set_ylabel('Magnitude')
            axs[0].legend()
            axs[0].set_xlim(0,500)

            peak1_freq_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak1_freq_pmt'].values[0]
            peak2_freq_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak2_freq_pmt'].values[0]
            peak3_freq_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'freq_3_pmt'].values[0]
            
            peak1_mag_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak1_mag_pmt'].values[0]
            peak2_mag_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'peak2_mag_pmt'].values[0]
            peak3_mag_pmt_value=stats_info_df.loc[stats_info_df['filename'] == filenames[i], 'mag_3_pmt'].values[0]
            
            # peak1_freq_index_pmt = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak1_freq_pmt_value, atol=1e-5))[0][0]
            # peak2_freq_index_pmt = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak2_freq_pmt_value, atol=1e-5))[0][0]
            # peak3_freq_index_pmt = np.where(np.isclose(inputsALL_fft[index_sample, :, 0], peak3_freq_pmt_value, atol=1e-5))[0][0]

            axs[1].plot(x,inputsALL_fft[index_sample, :, 4], label=f'Peak1 f:{peak1_freq_pmt_value:.2f} m:{peak1_mag_pmt_value:.2f}\n Peak2 f:{peak2_freq_pmt_value:.2f},m:{peak2_mag_pmt_value:.2f} \n Peak3 f:{peak3_freq_pmt_value:.2f},m:{peak3_mag_pmt_value:.2f}')
            # axs[1].plot(x[peak1_freq_index_pmt],inputsALL_fft[index_sample, peak1_freq_index_pmt, 4], 'ro')  # Plot the max pressure point
            # axs[1].plot(x[peak2_freq_index_pmt],inputsALL_fft[index_sample, peak2_freq_index_pmt, 4], 'bx')  # Plot the max pressure point 
            # axs[1].plot(x[peak3_freq_index_pmt],inputsALL_fft[index_sample, peak3_freq_index_pmt, 4], 'bx')  # Plot the max pressure point 
            
            axs[1].set_title('FFT PMT Signal')
            axs[1].set_xlabel('Frequency')
            axs[1].set_ylabel('Magnitude')
            axs[1].legend()
            axs[1].set_xlim(0,500)
            plt.suptitle(f"FFT Domain Features {filenames[i]}\n Stability: {stability_label_dict[filenames[i]]}\n New Cluster Label: {df['cluster_label'].iloc[i]}")
            plt.tight_layout()
            plt.show()