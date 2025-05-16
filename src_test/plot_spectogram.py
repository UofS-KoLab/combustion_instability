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
from scipy.signal import spectrogram
from scipy.signal import stft


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

def load_stability_labels(stability_filename):
    """Load stability labels from a CSV file."""
    stability_pd = pd.read_csv(stability_filename)
    logging.info(f"Found {len(stability_pd['Name'])} records in the labeling file.")
    
    stability_label_dict = {
        f"{row['Name']}": row['Stability']
        for _, row in stability_pd.iterrows()
    }
    return stability_label_dict


if __name__ == "__main__":
    args = parse_arguments()

    stability_label_dict = load_stability_labels(args.stability_file)
    names_files = list(stability_label_dict.keys())

    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    cache_file_fft = get_cache_file_path(args.project_root, args.window_size, "fft")
    inputsALL, outputsALL , output_name= get_data(cache_file)
    inputsALL_fft, outputsALL_fft, output_name_fft = get_data(cache_file_fft)
    
    print(inputsALL.shape)

    # choosen_file=["A2_Eq=0.49178","A2_Eq=0.31506","A2_Eq=0.68179", "A2_Eq=0.56601","A2_Eq=0.55553"]
    # choosen_file=["closed2_20kW_400slpm_0_","closed2_25kW_600slpm_0_","closed_20kW_400slpm_0_",
    #               "open_30kW_400slpm_40_","open_15kW_400slpm_10_",
    #               "open_20kW_400slpm_40_","open_25kW_600slpm_20_"]
    # Define the file path
    file_path = 'c:/Users/qpw475/Documents/combustion_instability/data/labels/spect_check.csv'

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Get the data from the 'filename' column into a list
    filename_list = df['filename'].tolist()

    # Print the list
    print(filename_list)
    # choosen_file=["closed2_20kW_400slpm_0_","closed2_25kW_600slpm_0_","closed2_20kW_400slpm_0_","closed2_25kW_600slpm_0_"]

    print("file", len(filename_list))

    for start_idx in range(0, len(filename_list), 16):
        choosen_file = filename_list[start_idx:start_idx + 16]
        # print(len(choosen_file))
        
        # choosen_file = filename_list[:16]

        # Create a 4x4 grid of subplots
        fig, axs = plt.subplots(4, 4, figsize=(15, 15))
        axs = axs.ravel()  # Flatten the 2D array of axes for easy iteration

        # Loop through each chosen file
        for idx, cho_file in enumerate(choosen_file):
            for i in range(inputsALL.shape[0]):
                if output_name[i] == cho_file:
                    pressure_signal = inputsALL[i, :, 0]
                    pmt_signal = inputsALL[i, :, 1]
                    fs = 1 / 0.0001

                    # Compute the STFT of both signals
                    f, t, Zxx_pressure = stft(pressure_signal, fs, window='hann', nperseg=256, noverlap=128)
                    f, t, Zxx_heat = stft(pmt_signal, fs, window='hann', nperseg=256, noverlap=128)

                    # Extract the phase of each signal
                    phase_pressure = np.angle(Zxx_pressure)  # Phase of pressure signal
                    phase_heat = np.angle(Zxx_heat)          # Phase of heat signal

                    # Compute the phase difference (in radians)
                    phase_difference = phase_heat - phase_pressure

                    # Wrap the phase difference to the range [-π, π]
                    phase_difference = np.angle(np.exp(1j * phase_difference))

                    # Plot the phase difference in the corresponding subplot
                    im = axs[idx].pcolormesh(t, f, phase_difference, shading='gouraud', cmap='hsv')
                    axs[idx].set_ylabel('Frequency [Hz]')
                    axs[idx].set_xlabel('Time [sec]')
                    # axs[idx].set_title(f'Phase Difference Between Pressure and Heat of {output_name[i]}')
                    axs[idx].set_title(f'{output_name[i]}')
                    axs[idx].set_ylim(5, 500)

        # Add a colorbar to the right of the grid
        cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Phase Difference [rad]')

        # Adjust layout to prevent overlap
        plt.tight_layout()
        plt.show()


























    # for cho_file in choosen_file:
    #     for i in range(inputsALL.shape[0]):
    #         if output_name[i]==cho_file:
    #             pressure_signal = inputsALL[i,:,0]
    #             pmt_signal = inputsALL[i,:,1]
    #             fs=1/0.0001
    #             # Compute the spectrogram
    #             f, t, Sxx = spectrogram(pressure_signal, fs, window='hann', nperseg=256, noverlap=128, scaling='spectrum')
    #             f_p, t_p, Sxx_p = spectrogram(pmt_signal, fs, window='hann', nperseg=256, noverlap=128, scaling='spectrum')


    #          # Compute the STFT
    #             f_pres_p, t_pres_p, Zxx_pres_p = stft(pressure_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase from the STFT
    #             phase = np.angle(Zxx_pres_p)  # Phase in radians

    #             # Compute the STFT
    #             f_pmt_p, t_pmt_p, Zxx_pmt_p = stft(pmt_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase from the STFT
    #             phase_pmt = np.angle(Zxx_pmt_p)  # Phase in radians

    #              #Compute the STFT of both signals
    #             f, t, Zxx_pressure = stft(pressure_signal, fs, window='hann', nperseg=256, noverlap=128)
    #             f, t, Zxx_heat = stft(pmt_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase of each signal
    #             phase_pressure = np.angle(Zxx_pressure)  # Phase of pressure signal
    #             phase_heat = np.angle(Zxx_heat)          # Phase of heat signal

    #             # Compute the phase difference (in radians)
    #             phase_difference = phase_heat - phase_pressure

    #             # Wrap the phase difference to the range [-π, π]
    #             phase_difference = np.angle(np.exp(1j * phase_difference))

    #             # Plot the phase difference
    #             plt.figure(figsize=(10, 6))
    #             plt.pcolormesh(t, f, phase_difference, shading='gouraud', cmap='hsv')
    #             plt.colorbar(label='Phase Difference [rad]')
    #             plt.ylabel('Frequency [Hz]')
    #             plt.xlabel('Time [sec]')
    #             plt.title(f'Phase Difference Between Pressure and Heat of {output_name[i]}')
    #             plt.ylim(5, 500)
    #             plt.show()












































    # for cho_file in choosen_file:
    #     for i in range(inputsALL.shape[0]):
    #         if output_name[i]==cho_file:
    #             pressure_signal = inputsALL[i,:,0]
    #             pmt_signal = inputsALL[i,:,1]
    #             fs=1/0.0001
    #             # Compute the spectrogram
    #             f, t, Sxx = spectrogram(pressure_signal, fs, window='hann', nperseg=256, noverlap=128, scaling='spectrum')
    #             f_p, t_p, Sxx_p = spectrogram(pmt_signal, fs, window='hann', nperseg=256, noverlap=128, scaling='spectrum')

    #             # # Plot the spectrogram
    #             # plt.figure(figsize=(10, 6))
    #             # plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')  # Convert to dB scale
    #             # plt.colorbar(label='Intensity (dB)')
    #             # plt.ylabel('Frequency [Hz]')
    #             # plt.xlabel('Time [sec]')
    #             # plt.title(f'Spectrogram Pressure of {output_name[i]} state: {stability_label_dict[output_name[i]]}')
    #             # plt.show()

    #             # # Plot the spectrogram
    #             # plt.figure(figsize=(10, 6))
    #             # plt.pcolormesh(t_p, f_p, 10 * np.log10(Sxx_p), shading='gouraud')  # Convert to dB scale
    #             # plt.colorbar(label='Intensity (dB)')
    #             # plt.ylabel('Frequency [Hz]')
    #             # plt.xlabel('Time [sec]')
    #             # plt.title(f'Spectrogram PMT of {output_name[i]} state: {stability_label_dict[output_name[i]]}')
    #             # plt.show()

    #             # Compute the STFT
    #             f_pres_p, t_pres_p, Zxx_pres_p = stft(pressure_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase from the STFT
    #             phase = np.angle(Zxx_pres_p)  # Phase in radians

    #             # # Plot the phase spectrogram
    #             # plt.figure(figsize=(10, 6))
    #             # plt.pcolormesh(t_pres_p, f_pres_p, phase, shading='gouraud', cmap='hsv')  # Use a cyclic colormap like 'hsv'
    #             # plt.colorbar(label='Phase [rad]')
    #             # plt.ylabel('Frequency [Hz]')
    #             # plt.xlabel('Time [sec]')
    #             # plt.title('Phase Spectrogram pressure')
    #             # plt.show()


    #             # Compute the STFT
    #             f_pmt_p, t_pmt_p, Zxx_pmt_p = stft(pmt_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase from the STFT
    #             phase_pmt = np.angle(Zxx_pmt_p)  # Phase in radians

    #             # # Plot the phase spectrogram
    #             # plt.figure(figsize=(10, 6))
    #             # plt.pcolormesh(t_pmt_p, f_pmt_p, phase_pmt, shading='gouraud', cmap='hsv')  # Use a cyclic colormap like 'hsv'
    #             # plt.colorbar(label='Phase [rad]')
    #             # plt.ylabel('Frequency [Hz]')
    #             # plt.xlabel('Time [sec]')
    #             # plt.title('Phase Spectrogram pmt')
    #             # plt.show()


    #             # # Plot the phase spectrogram
    #             # plt.figure(figsize=(10, 6))
    #             # plt.pcolormesh(t_pmt_p, f_pmt_p, phase-phase_pmt, shading='gouraud', cmap='hsv')  # Use a cyclic colormap like 'hsv'
    #             # plt.colorbar(label='Phase [rad]')
    #             # plt.ylabel('Frequency [Hz]')
    #             # plt.xlabel('Time [sec]')
    #             # plt.title(f'Phase Spectrogram pressure-pmt of {output_name[i]}')
    #             # plt.show()

    #             #Compute the STFT of both signals
    #             f, t, Zxx_pressure = stft(pressure_signal, fs, window='hann', nperseg=256, noverlap=128)
    #             f, t, Zxx_heat = stft(pmt_signal, fs, window='hann', nperseg=256, noverlap=128)

    #             # Extract the phase of each signal
    #             phase_pressure = np.angle(Zxx_pressure)  # Phase of pressure signal
    #             phase_heat = np.angle(Zxx_heat)          # Phase of heat signal

    #             # Compute the phase difference (in radians)
    #             phase_difference = phase_heat - phase_pressure

    #             # Wrap the phase difference to the range [-π, π]
    #             phase_difference = np.angle(np.exp(1j * phase_difference))

    #             # Plot the phase difference
    #             plt.figure(figsize=(10, 6))
    #             plt.pcolormesh(t, f, phase_difference, shading='gouraud', cmap='hsv')
    #             plt.colorbar(label='Phase Difference [rad]')
    #             plt.ylabel('Frequency [Hz]')
    #             plt.xlabel('Time [sec]')
    #             plt.title(f'Phase Difference Between Pressure and Heat of {output_name[i]}')
    #             plt.ylim(5, 500)
    #             plt.show()
