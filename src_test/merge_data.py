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

def get_old_data(old_data_file_path):
    # Split the columns argument into a list
    # columns_to_pick = ['filename','case','heat_input','flow_rate','hydrogen_ratio','stability','rms','rms_pmt']
    # csv_file_path = old_data_file_path
    # df = pd.read_csv(csv_file_path, usecols=columns_to_pick)
    # df['filename'] = df['filename'].str.split('.csv').str[0]
    df = pd.read_csv(old_data_file_path)    
    return df

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
    # print(inputsALL.shape)
    # print(outputsALL.shape)
    # print(outputsALL)
    return inputsALL,outputsALL, output_name
    # return inputsALL,outputsALL

def get_total_energy(y):
    # Calculate squared magnitude of FFT
    magnitude_squared = np.abs(y)**2

    # Sum to get total energy in frequency domain
    total_energy_freq = np.sum(magnitude_squared)

    # Normalize by N (if required)
    N = len(x)
    total_energy = total_energy_freq / N

    return total_energy

def extract_peak_custom_freq(x, y, range_low, range_high):

    frequencies_in_threshold = x[(x > range_low) & (x <= range_high)]
    y_values_in_threshold = y[(x > range_low) & (x <= range_high)]
    max_index=np.argmax(y_values_in_threshold)
    freq_max=frequencies_in_threshold[max_index]
    y_max=y[list(x).index(frequencies_in_threshold[max_index])]
    
    return freq_max,y_max

def extract_first_5_peaks(x, y, phase, y_pmt,phase_y_pmt):
    """
    Extract frequencies where the normalized y-values exceed a threshold.
    """
    # print("nothing")
    freq_p = x
    paired_p = list(zip(freq_p, y, phase))
    paired_sorted_p = sorted(paired_p, key=lambda x: x[1], reverse=True)
    f_p, mag_p, phase_p = zip(*paired_sorted_p)

    freq_pmt = x
    paired_pmt = list(zip(freq_pmt, y_pmt, phase_y_pmt))
    paired_sorted_pmt = sorted(paired_pmt, key=lambda x: x[1], reverse=True)
    f_pmt, mag_pmt, phase_pmt = zip(*paired_sorted_pmt)

    freq_1_y=f_p[0]
    mag_1_y=mag_p[0]
    phase_1_y=phase_p[0]

    freq_1_y_pmt=f_pmt[0]
    mag_1_y_pmt=mag_pmt[0]
    phase_1_y_pmt=phase_pmt[0]

    freq_2_y=f_p[1]
    mag_2_y=mag_p[1]
    phase_2_y=phase_p[1]
    
    freq_2_y_pmt=f_pmt[1]
    mag_2_y_pmt=mag_pmt[1]  
    phase_2_y_pmt=phase_pmt[1]

    freq_3_y=f_p[2]
    mag_3_y=mag_p[2]
    phase_3_y=phase_p[2]

    freq_3_y_pmt=f_pmt[2]
    mag_3_y_pmt=mag_pmt[2]
    phase_3_y_pmt=phase_pmt[2]

    freq_4_y=f_p[3]
    mag_4_y=mag_p[3]
    phase_4_y=phase_p[3]

    freq_4_y_pmt=f_pmt[3]
    mag_4_y_pmt=mag_pmt[3]
    phase_4_y_pmt=phase_pmt[3]

    freq_5_y=f_p[4]
    mag_5_y=mag_p[4]
    phase_5_y=phase_p[4]

    freq_5_y_pmt=f_pmt[4]
    mag_5_y_pmt=mag_pmt[4]
    phase_5_y_pmt=phase_pmt[4]

    phase_diff_1 = abs(np.degrees(phase_1_y) - np.degrees(phase_1_y_pmt))
    if phase_diff_1 > 180:
        phase_diff_1 = 360 - phase_diff_1

    phase_diff_2 = abs(np.degrees(phase_2_y) - np.degrees(phase_2_y_pmt))
    if phase_diff_2 > 180:
        phase_diff_2 = 360 - phase_diff_2

    phase_diff_3 = abs(np.degrees(phase_3_y) - np.degrees(phase_3_y_pmt))
    if phase_diff_3 > 180:
        phase_diff_3 = 360 - phase_diff_3
    
    phase_diff_4 = abs(np.degrees(phase_4_y) - np.degrees(phase_4_y_pmt))
    if phase_diff_4 > 180:
        phase_diff_4 = 360 - phase_diff_4

    phase_diff_5 = abs(np.degrees(phase_5_y) - np.degrees(phase_5_y_pmt))
    if phase_diff_5 > 180:
        phase_diff_5 = 360 - phase_diff_5

    return freq_1_y, mag_1_y, freq_1_y_pmt, mag_1_y_pmt, phase_diff_1, freq_2_y, mag_2_y, freq_2_y_pmt, mag_2_y_pmt, phase_diff_2, freq_3_y, mag_3_y, freq_3_y_pmt, mag_3_y_pmt, phase_diff_3, freq_4_y, mag_4_y, freq_4_y_pmt, mag_4_y_pmt, phase_diff_4, freq_5_y, mag_5_y, freq_5_y_pmt, mag_5_y_pmt, phase_diff_5
    
    

def extract_peak_range_phase(x, y,y_pmt,phase_p,phase_pmt, lower_bound=25, upper_bound=34, threshold=0.6):
    """
    Extract y-values where x-values are within the specified range.
    
    :param x: Array of x-values.
    :param y: Array of y-values.
    :param lower_bound: Lower bound of the x-value range.
    :param upper_bound: Upper bound of the x-value range.
    :return: Array of y-values within the specified x-value range.
    """
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_range=x[mask]
    y_values_in_range = y[mask]
    phase_y_values_in_range = phase_p[mask]
    max_index=np.argmax(y_values_in_range)
    max_y=y_values_in_range[max_index]
    x_max_p=x_range[max_index]
    phase_p_max=phase_y_values_in_range[max_index]

    y_pmt_values_in_range = y_pmt[mask]
    phase_y_pmt_values_in_range = phase_pmt[mask]
    max_index_pmt=np.argmax(y_pmt_values_in_range)
    max_y_pmt=y_pmt_values_in_range[max_index_pmt]
    x_max_pmt=x_range[max_index_pmt]
    phase_pmt_max=phase_y_pmt_values_in_range[max_index_pmt]

    phase_differece_value=abs(np.degrees(phase_p_max)-np.degrees(phase_pmt_max))
    if phase_differece_value>180:
        phase_differece_value=360-phase_differece_value

    scaler = MinMaxScaler()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    y_pmt_normalized = scaler.fit_transform(y_pmt.reshape(-1, 1)).flatten()
    frequencies_above_threshold_y = x[y_normalized > threshold]
    frequencies_above_threshold_y_pmt = x[y_pmt_normalized > threshold]
    common_frequencies = set(frequencies_above_threshold_y).intersection(frequencies_above_threshold_y_pmt)
    sync_freq_range = bool(len(common_frequencies)>0)


    return x_max_p, x_max_pmt, max_y, max_y_pmt,phase_differece_value,sync_freq_range


def extract_peak_range(x, y,y_pmt, lower_bound=25, upper_bound=34, threshold=0.6):
    """
    Extract y-values where x-values are within the specified range.
    
    :param x: Array of x-values.
    :param y: Array of y-values.
    :param lower_bound: Lower bound of the x-value range.
    :param upper_bound: Upper bound of the x-value range.
    :return: Array of y-values within the specified x-value range.
    """
    mask = (x >= lower_bound) & (x <= upper_bound)
    x_range=x[mask]
    y_values_in_range = y[mask]
    max_index=np.argmax(y_values_in_range)
    max_y=y_values_in_range[max_index]
    x_max=x_range[max_index]

    y_pmt_values_in_range = y_pmt[mask]
    max_y_pmt=y_pmt_values_in_range[max_index]

    scaler = MinMaxScaler()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()
    y_pmt_normalized = scaler.fit_transform(y_pmt.reshape(-1, 1)).flatten()
    frequencies_above_threshold_y = x[y_normalized > threshold]
    frequencies_above_threshold_y_pmt = x[y_pmt_normalized > threshold]
    common_frequencies = set(frequencies_above_threshold_y).intersection(frequencies_above_threshold_y_pmt)
    sync_freq_range = bool(common_frequencies.intersection(x_range))


    return x_max, max_y, max_y_pmt,sync_freq_range

def extract_peak_frequencies(x, y, phase, threshold=0.6):
    """
    Extract frequencies where the normalized y-values exceed a threshold.
    """
    # print(threshold)
    scaler = MinMaxScaler()
    y_normalized = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    frequencies_above_threshold = x[y_normalized > threshold]
    y_values_above_threshold = y_normalized[y_normalized > threshold]

    if len(y_values_above_threshold) > 0:
        max_index = np.argmax(y_values_above_threshold)
        freq_max=frequencies_above_threshold[max_index] 
        y_max=y[list(x).index(frequencies_above_threshold[max_index])]
        phase_value_peak_1=phase[list(x).index(frequencies_above_threshold[max_index])]

        index_max_norm = list(x).index(freq_max)

        y_normalized[index_max_norm] = -3000000
        second_max_index = np.argmax(y_normalized)
        freq_second_max = x[second_max_index]
        y_second_max = y[second_max_index]
        phase_value_peak_2=phase[second_max_index]

        return y_normalized,frequencies_above_threshold, y_values_above_threshold,freq_max,y_max,freq_second_max,y_second_max,phase_value_peak_1,phase_value_peak_2

    return [],[], [], None, None


def calculate_sync_score(pressure, pmt, freq, phase_p_in, phase_pmt_in, tolerance=2, total_peaks=10):
    fre_p = freq
    paired_pressure = list(zip(fre_p, pressure, phase_p_in))
    paired_pressure_sorted = sorted(paired_pressure, key=lambda x: x[1], reverse=True)
    f_p, mag_p, phase_p = zip(*paired_pressure_sorted)
    f_p=f_p[:total_peaks]
    mag_p=mag_p[:total_peaks]
    phase_p=phase_p[:total_peaks]

    freq_pmt = freq
    paired_pmt = list(zip(freq_pmt, pmt, phase_pmt_in))
    paired_pmt_sorted = sorted(paired_pmt, key=lambda x: x[1], reverse=True)
    f_pmt, mag_pmt, phase_pmt = zip(*paired_pmt_sorted)

    f_pmt=f_pmt[:total_peaks]
    mag_pmt=mag_pmt[:total_peaks]
    phase_pmt=phase_pmt[:total_peaks]

    sync_score = 0
    total_magnitude = 0

    print(f"Pressure peaks: {len(f_p)}")
    print(f"PMT peaks: {len(f_pmt)}")

    # Find the intersection of the two lists
    common_values = set(f_p).intersection(set(f_pmt))

    # Count the number of common values
    num_common_values = len(common_values)

    # Print the results
    print(f"Number of common values: {num_common_values}")
    print(f"Common values: {common_values}")

    new_score=num_common_values/len(f_pmt)

    for i in range(len(f_pmt)):
        # Find closest pressure peak within tolerance
        for j in range(len(f_p)):
            if abs(f_pmt[i] - f_p[j]) <= tolerance:
                # Calculate phase difference
                phase_diff = abs(phase_p[j] - phase_pmt[i])
                if phase_diff > 180:
                    phase_diff = 360 - phase_diff
                
                # Calculate weighted score
                sync_score += (mag_pmt[i] + mag_p[j]) * np.cos(np.radians(phase_diff))
                total_magnitude += (mag_pmt[i] + mag_p[j])
                break  # Move to next PMT peak after finding a match
        # Normalize the sync score
    if total_magnitude > 0:
        normalized_sync_score = sync_score / total_magnitude
    else:
        normalized_sync_score = 0

    return sync_score,normalized_sync_score, new_score




if __name__ == "__main__":
    args = parse_arguments()
    old_data=get_old_data(args.old_data_file)

    stability_label_dict = load_stability_labels(args.stability_file)

    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    inputsALL, outputsALL, output_name = get_data(cache_file)
    # inputsALL, outputsALL = get_data(cache_file)
    names_files = output_name

    # Initialize lists for results
    name, state = [], []
    peak1_freq_pressure,peak1_mag_pressure=[],[]
    peak1_freq_pmt, peak1_mag_pmt=[],[]
    peak2_freq_pressure,peak2_mag_pressure=[],[]
    peak2_freq_pmt, peak2_mag_pmt=[],[]
    sync_detected, sync_frequency, sync_mag_pressure, sync_mag_pmt, sync_total_peaks, sync_frequency_1st_peak = [], [], [], [], [], []
    total_peaks_pressure, total_peaks_pmt = [], []
    total_energy_p, total_energy_pmt = [], []

    ratio_frequency_peak_1, ratio_frequency_peak_2, ratio_frequency_peak_3=[],[],[]

    ration_amplitude_peak_1=[]

    phase_1_differece,phase_2_differece=[],[]
    phase_peak1_pressure, phase_peak2_pressure, phase_peak1_pmt, phase_peak2_pmt=[],[],[],[]
    norm_sync_score_first_10_peaks=[]
    norm_sync_score_first_5_peaks=[]

    norm_sync_score_range1_peaks=[]
    norm_sync_score_range2_peaks=[]

    peak_freq_5_100_p, peak_mag_5_100_p=[],[]
    peak_freq_5_100_pmt, peak_mag_5_100_pmt=[],[]
    peak_freq_100_200_p, peak_mag_100_200_p=[],[]
    peak_freq_100_200_pmt, peak_mag_100_200_pmt=[],[]
    peak_freq_200_300_p, peak_mag_200_300_p=[],[]
    peak_freq_200_300_pmt, peak_mag_200_300_pmt=[],[]
    peak_freq_300_400_p, peak_mag_300_400_p=[],[]
    peak_freq_300_400_pmt, peak_mag_300_400_pmt=[],[]
    peak_freq_400_500_p, peak_mag_400_500_p=[],[]
    peak_freq_400_500_pmt, peak_mag_400_500_pmt=[],[]

    freq_pressure_peak_range_1,peak_mag_pressure_range_1, peak_mag_p_pmt_range_1=[],[],[]
    freq_pressure_peak_range_2,peak_mag_pressure_range_2, peak_mag_p_pmt_range_2=[],[],[]
    sync_detected_range_1, sync_detected_range_2=[],[]

    highest_mag_p_rg1, highest_mag_pmt_rg1, highest_sync_rg1=[],[],[]
    highest_freq_p_rg1, highest_freq_pmt_rg1, phase_differece_rg1=[],[],[]

    freq_1_pressure,mag_1_pressure, freq_1_pmt,mag_1_pmt, phase_1_difference=[],[],[],[],[]
    freq_2_pressure,mag_2_pressure, freq_2_pmt,mag_2_pmt, phase_2_difference=[],[],[],[],[]
    freq_3_pressure,mag_3_pressure, freq_3_pmt,mag_3_pmt, phase_3_difference=[],[],[],[],[]
    freq_4_pressure,mag_4_pressure, freq_4_pmt,mag_4_pmt, phase_4_difference=[],[],[],[],[]
    freq_5_pressure,mag_5_pressure, freq_5_pmt,mag_5_pmt, phase_5_difference=[],[],[],[],[]

    score_instability_list=[]


    feature_names={"Frequency":0,"Pressure Magnitude":1,"Pressure Power":2,"Pressure Phase":3,
                   "PMT Magnitude":4,"PMT Power":5,"PMT Phase":6,
                   "Mult Magnitude":7,"Mult Power":8,"Phase Diff":9}
    freq_filter_threshold = 5 # Suppress values below 5Hz
    amplitude_threshold = 0.7  # Threshold for peak detection
    name_plot_p="Pressure Magnitude"
    name_plot_pmt="PMT Magnitude"
    print(inputsALL.shape)
    print(outputsALL.shape)
    # Process each sample
    for i in range(inputsALL.shape[0]):
    # for i in range(1):
        sample = inputsALL[i]

        x = sample[:, feature_names["Frequency"]]  # Frequencies
        y_p = sample[:, feature_names["Pressure Magnitude"]]  # Pressure magnitude
        y_p[x < freq_filter_threshold] = 0  # Suppress values below 5Hz
        phase_pressure= sample[:, feature_names["Pressure Phase"]]  # Pressure phase
        y_pmt = sample[:, feature_names["PMT Magnitude"]]  # PMT magnitude
        y_pmt[x < freq_filter_threshold] = 0  # Suppress values below 5Hz
        phase_pmt= sample[:, feature_names["PMT Phase"]]  # PMT phase

        values_peaks_5=extract_first_5_peaks(x, y_p, phase_pressure, y_pmt,phase_pmt)
        freq_1_pressure.append(values_peaks_5[0])
        mag_1_pressure.append(values_peaks_5[1])
        freq_1_pmt.append(values_peaks_5[2])
        mag_1_pmt.append(values_peaks_5[3])
        phase_1_difference.append(values_peaks_5[4])
        freq_2_pressure.append(values_peaks_5[5])
        mag_2_pressure.append(values_peaks_5[6])
        freq_2_pmt.append(values_peaks_5[7])
        mag_2_pmt.append(values_peaks_5[8])
        phase_2_difference.append(values_peaks_5[9])
        freq_3_pressure.append(values_peaks_5[10])
        mag_3_pressure.append(values_peaks_5[11])
        freq_3_pmt.append(values_peaks_5[12])
        mag_3_pmt.append(values_peaks_5[13])
        ratio_frequency_peak_3.append(values_peaks_5[10]/values_peaks_5[12])
        phase_3_difference.append(values_peaks_5[14])
        freq_4_pressure.append(values_peaks_5[15])
        mag_4_pressure.append(values_peaks_5[16])
        freq_4_pmt.append(values_peaks_5[17])
        mag_4_pmt.append(values_peaks_5[18])
        phase_4_difference.append(values_peaks_5[19])
        freq_5_pressure.append(values_peaks_5[20])
        mag_5_pressure.append(values_peaks_5[21])
        freq_5_pmt.append(values_peaks_5[22])
        mag_5_pmt.append(values_peaks_5[23])
        phase_5_difference.append(values_peaks_5[24])

        tolerance = 5  # e = 5

        # Check if freq_1_pressure and freq_1_pmt are close
        first_peak_check=0
        if abs(values_peaks_5[0] - values_peaks_5[2]) <= tolerance:
            # print(f"Sample {i+1}: freq_1_pressure ({freq_1_pressure[i]:.2f} Hz) and freq_1_pmt ({freq_1_pmt[i]:.2f} Hz) are close.")
            sync_frequency_1st_peak.append(1)
            first_peak_check=1
        else:
            sync_frequency_1st_peak.append(0)
            first_peak_check=0
            # print(f"Sample {i+1}: freq_1_pressure ({freq_1_pressure[i]:.2f} Hz) and freq_1_pmt ({freq_1_pmt[i]:.2f} Hz) are NOT close.")



        # f=extract_peak_range(x, y_p,y_pmt, lower_bound=25, upper_bound=34)
        f_p_rg_1, mag_p_rg_1, mag_pmt_rg_1, sync_d_rg_1 = extract_peak_range(
            x, y_p, y_pmt, lower_bound=25, upper_bound=34,threshold=amplitude_threshold
        )
        f_p_rg_2, mag_p_rg_2, mag_pmt_rg_2, sync_d_rg_2 = extract_peak_range(
            x, y_p, y_pmt, lower_bound=132, upper_bound=144,threshold=amplitude_threshold
        )

        freq_pressure_peak_range_1.append(f_p_rg_1)
        peak_mag_pressure_range_1.append(mag_p_rg_1)
        peak_mag_p_pmt_range_1.append(mag_pmt_rg_1)
        sync_detected_range_1.append(sync_d_rg_1)

        freq_pressure_peak_range_2.append(f_p_rg_2)
        peak_mag_pressure_range_2.append(mag_p_rg_2)
        peak_mag_p_pmt_range_2.append(mag_pmt_rg_2)
        sync_detected_range_2.append(sync_d_rg_2)

        freq_r1_p, mag_r1_p=extract_peak_custom_freq(x,y_p,5,100)
        freq_r1_pmt, mag_r1_pmt=extract_peak_custom_freq(x,y_pmt,5,100)
        freq_r2_p, mag_r2_p=extract_peak_custom_freq(x,y_p,100,200)
        freq_r2_pmt, mag_r2_pmt=extract_peak_custom_freq(x,y_pmt,100,200)
        freq_r3_p, mag_r3_p=extract_peak_custom_freq(x,y_p,200,300)
        freq_r3_pmt, mag_r3_pmt=extract_peak_custom_freq(x,y_pmt,200,300)
        freq_r4_p, mag_r4_p=extract_peak_custom_freq(x,y_p,300,400)
        freq_r4_pmt, mag_r4_pmt=extract_peak_custom_freq(x,y_pmt,300,400)
        freq_r5_p, mag_r5_p=extract_peak_custom_freq(x,y_p,400,500)
        freq_r5_pmt, mag_r5_pmt=extract_peak_custom_freq(x,y_pmt,400,500)

        peak_freq_5_100_p.append(freq_r1_p)
        peak_mag_5_100_p.append(mag_r1_p)
        peak_freq_5_100_pmt.append(freq_r1_pmt)
        peak_mag_5_100_pmt.append(mag_r1_pmt)
        peak_freq_100_200_p.append(freq_r2_p)
        peak_mag_100_200_p.append(mag_r2_p)
        peak_freq_100_200_pmt.append(freq_r2_pmt)
        peak_mag_100_200_pmt.append(mag_r2_pmt)
        peak_freq_200_300_p.append(freq_r3_p)
        peak_mag_200_300_p.append(mag_r3_p)
        peak_freq_200_300_pmt.append(freq_r3_pmt)
        peak_mag_200_300_pmt.append(mag_r3_pmt)
        peak_freq_300_400_p.append(freq_r4_p)
        peak_mag_300_400_p.append(mag_r4_p)
        peak_freq_300_400_pmt.append(freq_r4_pmt)
        peak_mag_300_400_pmt.append(mag_r4_pmt)
        peak_freq_400_500_p.append(freq_r5_p)
        peak_mag_400_500_p.append(mag_r5_p)
        peak_freq_400_500_pmt.append(freq_r5_pmt)
        peak_mag_400_500_pmt.append(mag_r5_pmt)

        total_energy_p_val=get_total_energy(y_p)
        total_energy_pmt_val=get_total_energy(y_pmt)

        total_energy_p.append(total_energy_p_val)
        total_energy_pmt.append(total_energy_pmt_val)
        # Extract peaks for both signals
        (
            y_p_normalized,
            freq_p,
            mag_p,
            max_freq_p,
            max_real_p_val,
            max2_freq_p,
            max2_real_p_val,
            phase_p_1,
            phase_p_2,
        ) = extract_peak_frequencies(x, y_p, phase_pressure, amplitude_threshold)
        (
            y_pmt_normalized,
            freq_pmt,
            mag_pmt,
            max_freq_pmt,
            max_real_pmt_val,
            max2_freq_pmt,
            max2_real_pmt_val,
            phase_pmt_1,
            phase_pmt_2,
        ) = extract_peak_frequencies(x, y_pmt, phase_pmt, amplitude_threshold)

        
        phase_1_differece_value=abs(np.degrees(phase_p_1)-np.degrees(phase_pmt_1))
        phase_2_differece_value=abs(np.degrees(phase_p_2)-np.degrees(phase_pmt_2))
        if phase_1_differece_value>180:
            phase_1_differece_value=360-phase_1_differece_value
        if phase_2_differece_value>180:
            phase_2_differece_value=360-phase_2_differece_value
        phase_1_differece.append(phase_1_differece_value)
        phase_2_differece.append(phase_2_differece_value)
        phase_peak1_pressure.append(np.degrees(phase_p_1))
        phase_peak2_pressure.append(np.degrees(phase_p_2))
        phase_peak1_pmt.append(np.degrees(phase_pmt_1))
        phase_peak2_pmt.append(np.degrees(phase_pmt_2))


        
        
        
        norm_sync_score_first_10_peaks.append(calculate_sync_score(y_p,y_pmt,x,phase_pressure,phase_pmt,2,10)[0])

        norm_sync_score_first_5_peaks.append(calculate_sync_score(y_p,y_pmt,x,phase_pressure,phase_pmt,2,3)[0])


        score_instability=0
        new_score_10=calculate_sync_score(y_p,y_pmt,x,phase_pressure,phase_pmt,2,10)[2]
        if first_peak_check==1:
            score_instability=1
        else:
            score_instability=new_score_10

        score_instability_list.append(score_instability)
        
        mask = (x >= 25) & (x <= 34)
        x_range=x[mask]
        y_p_range=y_p[mask]
        y_pmt_values_in_range = y_pmt[mask]
        norm_sync_score_range1_peaks.append(calculate_sync_score(y_p_range,y_pmt_values_in_range,x_range,phase_pressure,phase_pmt,2,10)[0])


        mask = (x >= 132) & (x <= 144)
        x_range=x[mask]
        y_p_range=y_p[mask]
        y_pmt_values_in_range = y_pmt[mask]
        norm_sync_score_range2_peaks.append(calculate_sync_score(y_p_range,y_pmt_values_in_range,x_range,phase_pressure,phase_pmt,2,10)[0])


        x_max_p_rg1_value, x_max_pmt_rg1_value, max_y_rg1, max_y_pmt_rg1,phase_differece_value_rg1,sync_freq_rg1_by_len=extract_peak_range_phase(x, y_p,y_pmt,phase_pressure,phase_pmt, 25, 34, 0.7)
        highest_mag_p_rg1.append(max_y_rg1)
        highest_mag_pmt_rg1.append(max_y_pmt_rg1)
        
        highest_freq_p_rg1.append(x_max_p_rg1_value)
        highest_freq_pmt_rg1.append(x_max_pmt_rg1_value)

        highest_sync_rg1.append(sync_freq_rg1_by_len)
        phase_differece_rg1.append(phase_differece_value_rg1)

        
        
        # Store results
        name.append(names_files[i])
        state.append(stability_label_dict[names_files[i]])
        # name.append(outputsALL[i])
        # state.append(outputsALL[i])
        peak1_freq_pressure.append(max_freq_p)
        peak1_mag_pressure.append(max_real_p_val)
        peak1_freq_pmt.append(max_freq_pmt)
        peak1_mag_pmt.append(max_real_pmt_val)

        ratio_frequency_peak_1.append(max_freq_p/max_freq_pmt)
        ration_amplitude_peak_1.append(max_real_p_val/max_real_pmt_val)
        

        peak2_freq_pressure.append(max2_freq_p)
        peak2_mag_pressure.append(max2_real_p_val)
        peak2_freq_pmt.append(max2_freq_pmt)
        peak2_mag_pmt.append(max2_real_pmt_val)

        ratio_frequency_peak_2.append(max2_freq_p/max2_freq_pmt)

        total_peaks_pressure.append(len(freq_p))
        total_peaks_pmt.append(len(freq_pmt))

        # Identify common frequencies
        common_frequencies = set(freq_p).intersection(freq_pmt)
        pressure_sync_magnitudes=[y_p[list(x).index(freq)] for freq in list(common_frequencies)]
        pmt_sum_sync_magnitudes=[y_pmt[list(x).index(freq)] for freq in list(common_frequencies)]
        filename_temporal = [names_files[i]] * len(common_frequencies)
        stability_temporal = [stability_label_dict[names_files[i]]] * len(common_frequencies)
        # filename_temporal = [outputsALL[i]] * len(common_frequencies)
        # stability_temporal = [outputsALL[i]] * len(common_frequencies)

        # # Create a new DataFrame
        # data_temporal = {
        #     'filename': filename_temporal,
        #     'stability': stability_temporal,
        #     'threshold': [amplitude_threshold] * len(common_frequencies),
        #     'frequency': list(common_frequencies),
        #     'pressure_magnitude': pressure_sync_magnitudes,
        #     'pmt_magnitude': pmt_sum_sync_magnitudes
        # }
        # new_df = pd.DataFrame(data_temporal)

        # # Print the new DataFrame
        # print(new_df)

        # # Save the new DataFrame to a CSV file
        # output_csv_path = os.path.join("C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\stats",f"sync_values_{amplitude_threshold}.csv")
        # # Check if the file already exists
        # if not os.path.isfile(output_csv_path):
        #     # If the file does not exist, write the DataFrame with the header
        #     new_df.to_csv(output_csv_path, index=False)
        # else:
        #     # If the file exists, append the DataFrame without writing the header
        #     new_df.to_csv(output_csv_path, mode='a', header=False, index=False)

        # print(f"New DataFrame saved to {output_csv_path}")

        sync_detected_bool=len(common_frequencies) > 0

        sync_detected.append(sync_detected_bool)

        if sync_detected_bool:
            pressure_mag_sync=[y_p[list(x).index(freq)] for freq in list(common_frequencies)]
            sync_mag_pressure_val=np.max(pressure_mag_sync)
            sync_freq_val=x[list(y_p).index(sync_mag_pressure_val)]
            sync_mag_pmt_val=y_pmt[list(x).index(sync_freq_val)]
            sync_total_peaks_val=len(common_frequencies)

        else:
            # sync_freq_val=max_freq_p
            # sync_mag_pressure_val=max_real_p_val
            # sync_mag_pmt_val=y_pmt[list(x).index(max_freq_p)]
            sync_freq_val=-1
            sync_mag_pressure_val=-1
            sync_mag_pmt_val=-1
            sync_total_peaks_val=0

        sync_frequency.append(sync_freq_val)
        sync_mag_pressure.append(sync_mag_pressure_val)
        sync_mag_pmt.append(sync_mag_pmt_val)
        sync_total_peaks.append(len(common_frequencies))

    print(len(name))

    # Create DataFrame
    df_new = pd.DataFrame({
        "filename": name,
        "stability": state,
        "peak1_freq_pressure": peak1_freq_pressure,
        "peak1_mag_pressure": peak1_mag_pressure,
        "peak1_freq_pmt": peak1_freq_pmt,
        "peak1_mag_pmt": peak1_mag_pmt,
        "peak2_freq_pressure": peak2_freq_pressure,
        "peak2_mag_pressure": peak2_mag_pressure,
        "peak2_freq_pmt": peak2_freq_pmt,
        "peak2_mag_pmt": peak2_mag_pmt,
        "ratio_freq_1": ratio_frequency_peak_1,
        "ratio_freq_2": ratio_frequency_peak_2,
        "ratio_freq_3": ratio_frequency_peak_3,
        "ratio_amplitude_1":ration_amplitude_peak_1,

        "sync_detected_range_1": sync_detected_range_1,
        "freq_pressure_peak_range_1": freq_pressure_peak_range_1,
        "peak_mag_pressure_range_1": peak_mag_pressure_range_1,
        "peak_mag_p_pmt_range_1": peak_mag_p_pmt_range_1,
        "sync_detected_range_2": sync_detected_range_2,
        "freq_pressure_peak_range_2": freq_pressure_peak_range_2,
        "peak_mag_pressure_range_2": peak_mag_pressure_range_2,
        "peak_mag_p_pmt_range_2": peak_mag_p_pmt_range_2,
        
        "phase_1_differece": phase_1_differece,
        "phase_2_differece": phase_2_differece,
        "phase_peak1_pressure": phase_peak1_pressure,
        "phase_peak2_pressure": phase_peak2_pressure,
        "phase_peak1_pmt": phase_peak1_pmt,
        "phase_peak2_pmt": phase_peak2_pmt,

        "norm_sync_score_first_10_peaks": norm_sync_score_first_10_peaks,
        "norm_sync_score_first_5_peaks": norm_sync_score_first_5_peaks,
        "norm_sync_score_range1_peaks": norm_sync_score_range1_peaks,
        "norm_sync_score_range2_peaks": norm_sync_score_range2_peaks,

        "highest_mag_p_rg1": highest_mag_p_rg1,
        "highest_mag_pmt_rg1": highest_mag_pmt_rg1,
        "highest_freq_p_rg1": highest_freq_p_rg1,
        "highest_freq_pmt_rg1": highest_freq_pmt_rg1,
        "highest_sync_rg1": highest_sync_rg1,
        "phase_differece_rg1": phase_differece_rg1,

        "freq_1_pressure": freq_1_pressure,
        "mag_1_pressure": mag_1_pressure,
        "freq_1_pmt": freq_1_pmt,
        "mag_1_pmt": mag_1_pmt,
        "phase_1_difference": phase_1_difference,
        "freq_2_pressure": freq_2_pressure,
        "mag_2_pressure": mag_2_pressure,
        "freq_2_pmt": freq_2_pmt,
        "mag_2_pmt": mag_2_pmt,
        "phase_2_difference": phase_2_difference,
        "freq_3_pressure": freq_3_pressure,
        "mag_3_pressure": mag_3_pressure,
        "freq_3_pmt": freq_3_pmt,
        "mag_3_pmt": mag_3_pmt,
        "phase_3_difference": phase_3_difference,
        "freq_4_pressure": freq_4_pressure,
        "mag_4_pressure": mag_4_pressure,
        "freq_4_pmt": freq_4_pmt,
        "mag_4_pmt": mag_4_pmt,
        "phase_4_difference": phase_4_difference,
        "freq_5_pressure": freq_5_pressure,
        "mag_5_pressure": mag_5_pressure,
        "freq_5_pmt": freq_5_pmt,
        "mag_5_pmt": mag_5_pmt,
        "phase_5_difference": phase_5_difference,

        "sync_frequency_1st_peak": sync_frequency_1st_peak,
        "score_instability":score_instability_list,


        "sync_detected": sync_detected,
        # "sync_frequency": sync_frequency,
        # "sync_mag_pressure": sync_mag_pressure,
        # "sync_mag_pmt": sync_mag_pmt,
        "sync_total_peaks": sync_total_peaks,
        # "total_peaks_pressure": total_peaks_pressure,
        # "total_peaks_pmt": total_peaks_pmt,
        # "total_energy_p": total_energy_p,
        # "total_energy_pmt": total_energy_pmt,
        # "peak_freq_5_100_p": peak_freq_5_100_p,
        # "peak_mag_5_100_p": peak_mag_5_100_p,
        # "peak_freq_5_100_pmt": peak_freq_5_100_pmt,
        # "peak_mag_5_100_pmt": peak_mag_5_100_pmt,
        # "peak_freq_100_200_p": peak_freq_100_200_p,
        # "peak_mag_100_200_p": peak_mag_100_200_p,
        # "peak_freq_100_200_pmt": peak_freq_100_200_pmt,
        # "peak_mag_100_200_pmt": peak_mag_100_200_pmt,
        # "peak_freq_200_300_p": peak_freq_200_300_p,
        # "peak_mag_200_300_p": peak_mag_200_300_p,
        # "peak_freq_200_300_pmt": peak_freq_200_300_pmt,
        # "peak_mag_200_300_pmt": peak_mag_200_300_pmt,
        # "peak_freq_300_400_p": peak_freq_300_400_p,
        # "peak_mag_300_400_p": peak_mag_300_400_p,
        # "peak_freq_300_400_pmt": peak_freq_300_400_pmt,
        # "peak_mag_300_400_pmt": peak_mag_300_400_pmt,
        # "peak_freq_400_500_p": peak_freq_400_500_p,
        # "peak_mag_400_500_p": peak_mag_400_500_p,
        # "peak_freq_400_500_pmt": peak_freq_400_500_pmt,
        # "peak_mag_400_500_pmt": peak_mag_400_500_pmt,

    })

    # Find columns in df_new that do not exist in old_data
    columns_to_merge = [col for col in df_new.columns if col not in old_data.columns]

    # Merge old_data with df_new based on the filename column
    merged_data = pd.merge(old_data, df_new[['filename'] + columns_to_merge], on='filename', how='left')
    # merged_data = pd.merge(old_data, df_new[['filename'] + columns_to_merge], on='filename', how='left')
   
    # df_new=df_new.drop(columns=["filename","stability"])
    # merged_data = pd.concat([old_data, df_new], axis=1)
    # print(merged_data)

    merge_data_path=os.path.join(args.project_root, "data", "cluster")
    os.makedirs(merge_data_path, exist_ok=True)
    # Save the merged DataFrame to a CSV file
    output_csv_path = os.path.join(merge_data_path,f"{args.fuel_type}_cluster_thr_{amplitude_threshold}.csv")  # Replace with the desired output path
    merged_data.to_csv(output_csv_path, index=False)

    print(f"Merged data saved to {output_csv_path}")


# Save the DataFrame to a CSV file
output_csv_path = "output2.csv"  # Replace with the desired output path
df_new.to_csv(output_csv_path, index=False)
