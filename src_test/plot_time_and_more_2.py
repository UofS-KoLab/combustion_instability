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
from scipy.signal import find_peaks
from scipy.signal import stft
from scipy.stats import linregress

DATA_FILE="C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\raw\\transient\\hidrogen\\open_25kW_600slpm_80%to70%.csv"

# Load the CSV file into a DataFrame
df = pd.read_csv(DATA_FILE)

# Define the column names
df.columns = ['time', 'pressure', 'heat']

# Print the first few rows of the DataFrame to verify the column names
print("DataFrame with Defined Columns:")
print(df.head())

# Print the first row values
first_row = df.iloc[0]
print("First Row Values:")
print(first_row)
first_row = df.iloc[1]
print("First Row Values:")
print(first_row)

# Define the time column to go from 0 to 24 seconds
df['time'] = np.linspace(0, 24, len(df))

# Print the first few rows to verify the time column
print("DataFrame with Updated Time Column:")
print(df.head())

# Plot each column with the updated time column as the x-axis
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create subplots for pressure and heat

# Plot pressure
axs[0].plot(df['time'], df['pressure'], label='Pressure')
axs[0].axvline(x=5.6, color='red', linestyle='--', linewidth=1, label='5.6 seconds')
axs[0].set_title('Pressure')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Pressure')
axs[0].legend()

# Plot heat
axs[1].plot(df['time'], df['heat'], label='Heat', color='orange')
axs[1].axvline(x=5.6, color='red', linestyle='--', linewidth=1, label='5.6 seconds')
axs[1].set_title('Heat')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Heat')
axs[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()









# Filter the DataFrame for the time range from 0 to 18 seconds
filtered_df_0_to_4 = df[(df['time'] >= 0) & (df['time'] <= 4)]

# Calculate the standard deviation for pressure and heat
std_pressure = filtered_df_0_to_4['pressure'].std()
std_heat = filtered_df_0_to_4['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure = filtered_df_0_to_4['pressure'].max() - filtered_df_0_to_4['pressure'].min()
ptp_heat = filtered_df_0_to_4['heat'].max() - filtered_df_0_to_4['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (0 to 4 seconds): {std_pressure}")
print(f"Standard Deviation of Heat (0 to 4 seconds): {std_heat}")
print(f"Peak-to-Peak of Pressure (0 to 4 seconds): {ptp_pressure}")
print(f"Peak-to-Peak of Heat (0 to 4 seconds): {ptp_heat}")

# Filter the DataFrame for the time range from 18 to 19.3 seconds
filtered_df_18_to_19_3 = df[(df['time'] >= 4) & (df['time'] <= 5.6)]

# Calculate the standard deviation for pressure and heat
std_pressure_18_to_19_3 = filtered_df_18_to_19_3['pressure'].std()
std_heat_18_to_19_3 = filtered_df_18_to_19_3['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure_18_to_19_3 = filtered_df_18_to_19_3['pressure'].max() - filtered_df_18_to_19_3['pressure'].min()
ptp_heat_18_to_19_3 = filtered_df_18_to_19_3['heat'].max() - filtered_df_18_to_19_3['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (4 to 5.6 seconds): {std_pressure_18_to_19_3}")
print(f"Standard Deviation of Heat (4 to 5.6 seconds): {std_heat_18_to_19_3}")
print(f"Peak-to-Peak of Pressure (4 to 5.6 seconds): {ptp_pressure_18_to_19_3}")
print(f"Peak-to-Peak of Heat (4 to 5.6 seconds): {ptp_heat_18_to_19_3}")

# Filter the DataFrame for the time range from 19.3 to 20 seconds
filtered_df_19_3_to_20 = df[(df['time'] >5.6) & (df['time'] <= 6.6)]

# Calculate the standard deviation for pressure and heat
std_pressure_19_3_to_20 = filtered_df_19_3_to_20['pressure'].std()
std_heat_19_3_to_20 = filtered_df_19_3_to_20['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure_19_3_to_20 = filtered_df_19_3_to_20['pressure'].max() - filtered_df_19_3_to_20['pressure'].min()
ptp_heat_19_3_to_20 = filtered_df_19_3_to_20['heat'].max() - filtered_df_19_3_to_20['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (5.6 to 6.6 seconds): {std_pressure_19_3_to_20}")
print(f"Standard Deviation of Heat (5.6 to 6.6 seconds): {std_heat_19_3_to_20}")
print(f"Peak-to-Peak of Pressure (5.6 to 6.6 seconds): {ptp_pressure_19_3_to_20}")
print(f"Peak-to-Peak of Heat (5.6 to 6.6 seconds): {ptp_heat_19_3_to_20}")

# Filter the DataFrame for the time range from 17 to 18 seconds
filtered_df_17_to_18 = df[(df['time'] >= 6.6) & (df['time'] <= 15)]

# Calculate the standard deviation for pressure and heat
std_pressure_17_to_18 = filtered_df_17_to_18['pressure'].std()
std_heat_17_to_18 = filtered_df_17_to_18['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure_17_to_18 = filtered_df_17_to_18['pressure'].max() - filtered_df_17_to_18['pressure'].min()
ptp_heat_17_to_18 = filtered_df_17_to_18['heat'].max() - filtered_df_17_to_18['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (6.6 to 15 seconds): {std_pressure_17_to_18}")
print(f"Standard Deviation of Heat (6.6 to 15 seconds): {std_heat_17_to_18}")
print(f"Peak-to-Peak of Pressure (6.6 to 15 seconds): {ptp_pressure_17_to_18}")
print(f"Peak-to-Peak of Heat (6.6 to 15 seconds): {ptp_heat_17_to_18}")

# Filter the DataFrame for the time range from 20 to 21 seconds
filtered_df_20_to_21 = df[(df['time'] >= 15) & (df['time'] <= 24)]

# Calculate the standard deviation for pressure and heat
std_pressure_20_to_21 = filtered_df_20_to_21['pressure'].std()
std_heat_20_to_21 = filtered_df_20_to_21['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure_20_to_21 = filtered_df_20_to_21['pressure'].max() - filtered_df_20_to_21['pressure'].min()
ptp_heat_20_to_21 = filtered_df_20_to_21['heat'].max() - filtered_df_20_to_21['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (15 to 24 seconds): {std_pressure_20_to_21}")
print(f"Standard Deviation of Heat (15 to 24 seconds): {std_heat_20_to_21}")
print(f"Peak-to-Peak of Pressure (15 to 24 seconds): {ptp_pressure_20_to_21}")
print(f"Peak-to-Peak of Heat (15 to 24 seconds): {ptp_heat_20_to_21}")



# Filter the DataFrame for the time range from 20 to 24 seconds
filtered_df_20_to_24 = df[(df['time'] >= 6.6) & (df['time'] <= 7.6)]

# Calculate the standard deviation for pressure and heat
std_pressure_20_to_24 = filtered_df_20_to_24['pressure'].std()
std_heat_20_to_24 = filtered_df_20_to_24['heat'].std()

# Calculate the peak-to-peak values for pressure and heat
ptp_pressure_20_to_24 = filtered_df_20_to_24['pressure'].max() - filtered_df_20_to_24['pressure'].min()
ptp_heat_20_to_24 = filtered_df_20_to_24['heat'].max() - filtered_df_20_to_24['heat'].min()

# Print the results
print(f"Standard Deviation of Pressure (6.6 to 7.6 seconds): {std_pressure_20_to_24}")
print(f"Standard Deviation of Heat (6.6 to 7.6 seconds): {std_heat_20_to_24}")
print(f"Peak-to-Peak of Pressure (6.6 to 7.6 seconds): {ptp_pressure_20_to_24}")
print(f"Peak-to-Peak of Heat (6.6 to 7.6 seconds): {ptp_heat_20_to_24}")
















# Filter the DataFrame for the time range from 18 to 20 seconds
filtered_df = df[(df['time'] >= 4.6) & (df['time'] <= 6.6)]

# Create a new figure for the filtered data
fig, axs = plt.subplots(2, 1, figsize=(10, 8))  # Create subplots for pressure and heat

# Plot pressure
axs[0].plot(filtered_df['time'], filtered_df['pressure'], label='Pressure')
axs[0].axvline(x=5.6, color='red', linestyle='--', linewidth=1, label='5.6 seconds')
axs[0].set_title('Pressure (4.6 to 5.6 seconds)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Pressure')
axs[0].legend()

# Plot heat
axs[1].plot(filtered_df['time'], filtered_df['heat'], label='Heat', color='orange')
axs[1].axvline(x=5.6, color='red', linestyle='--', linewidth=1, label='5.6 seconds')
axs[1].set_title('Heat (4.6 to 5.6 seconds)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Heat')
axs[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()


from scipy.fft import fft, fftfreq
# Convert filtered_df['pressure'] and filtered_df['heat'] to lists
pressure_list = filtered_df['pressure'].tolist()
heat_list = filtered_df['heat'].tolist()

# Compute the FFT for pressure and heat
fs = 1 / 0.0001  # Sampling frequency
pressure_fft = fft(pressure_list)
heat_fft = fft(heat_list)


# Compute the frequency bins
freqs = fftfreq(len(filtered_df), d=1/fs)

# Filter only the positive frequencies
positive_freqs = freqs[:len(freqs)//2]
pressure_fft_magnitude = np.abs(pressure_fft[:len(freqs)//2])
heat_fft_magnitude = np.abs(heat_fft[:len(freqs)//2])


# Apply the 500 Hz filter
freq_filter = positive_freqs <= 500
filtered_freqs = positive_freqs[freq_filter]
filtered_pressure_fft = pressure_fft_magnitude[freq_filter]
filtered_heat_fft = heat_fft_magnitude[freq_filter]
filtered_heat_fft[filtered_freqs < 5] = 0

# Plot the FFT results
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot FFT of pressure
axs[0].plot(filtered_freqs, filtered_pressure_fft, label='Pressure FFT')
axs[0].set_title('FFT of Pressure (4.6 to 6.6 seconds, up to 500 Hz)')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Magnitude')
axs[0].legend()

# Plot FFT of heat
axs[1].plot(filtered_freqs, filtered_heat_fft, label='Heat FFT', color='orange')
axs[1].set_title('FFT of Heat (4.6 to 6.6 seconds, up to 500 Hz)')
axs[1].set_xlabel('Frequency (Hz)')
axs[1].set_ylabel('Magnitude')
axs[1].legend()

# Adjust layout and show the plot
plt.tight_layout()
plt.show()














inicio=10.6
final=24
filtered_df_spec = df[(df['time'] >= inicio) & (df['time'] <= final)]
pressure_list_spec = filtered_df_spec['pressure'].tolist()
heat_list_spec = filtered_df_spec['heat'].tolist()

# Compute the STFT of both signals
f, t, Zxx_pressure = stft(pressure_list_spec, fs, window='hann', nperseg=256, noverlap=128)
f, t, Zxx_heat = stft(heat_list_spec, fs, window='hann', nperseg=256, noverlap=128)

# Extract the phase of each signal
phase_pressure = np.angle(Zxx_pressure)  # Phase of pressure signal
phase_heat = np.angle(Zxx_heat)          # Phase of heat signal

# Compute the phase difference (in radians)
phase_difference = phase_heat - phase_pressure

# Wrap the phase difference to the range [-π, π]
phase_difference = np.angle(np.exp(1j * phase_difference))

# Plot the phase difference in the corresponding subplot
fig, axs = plt.subplots(1, 1, figsize=(10, 8))  # Create a new figure

im = axs.pcolormesh(t, f, phase_difference, shading='gouraud', cmap='hsv')
axs.set_ylabel('Frequency [Hz]')
axs.set_xlabel('Time [sec]')
axs.set_title(f'Phase Difference Between Pressure and Heat ({inicio}s to {final}s)')
axs.set_ylim(5, 500)
# axs.set_xlim(15, 24)
# Add a colorbar
cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
cbar.set_label('Phase Difference [rad]')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()
plt.close(fig)  # Explicitly close the figure

# Select specific frequency bands to plot as lines
selected_frequencies = [40,110,130,140]  # Frequencies to plot
line_data = {}

# Find the indices of the selected frequencies
phase_difference = np.abs(phase_difference)

for freq in selected_frequencies:
    freq_idx = np.argmin(np.abs(f - freq))  # Find the closest frequency index
    line_data[freq] = phase_difference[freq_idx, :]  # Extract the phase difference for this frequency

# Plot the phase difference as line plots for the selected frequencies
plt.figure(figsize=(10, 6))
for freq, data in line_data.items():
    plt.plot(t, data, label=f'{freq} Hz')

plt.xlabel('Time [sec]')
plt.ylabel('Phase Difference [rad]')
plt.title('Phase Difference Over Time for Selected Frequencies')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# Measure the trend for each selected frequency
for freq in selected_frequencies:
    freq_idx = np.argmin(np.abs(f - freq))  # Find the closest frequency index
    data = phase_difference[freq_idx, :]  # Extract the phase difference for this frequency

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(t, data)

    print(f"Frequency: {freq} Hz")
    print(f"  Slope: {slope:.4f}")
    if slope < 0:
        print("  The trend is decreasing.")
    else:
        print("  The trend is increasing or constant.")
    print()







# Define the sampling frequency and the duration of each segment (100ms)
fs = 1 / 0.0001  # Sampling frequency (10,000 Hz)
segment_duration = 0.1  # 100ms in seconds
samples_per_segment = int(segment_duration * fs)  # Number of samples in 100ms

# Split the data into chunks of 100ms
num_segments = len(pressure_list) // samples_per_segment

# Iterate over each segment and compute the FFT
for i in range(num_segments):
    # Extract the segment for pressure and heat
    start_idx = i * samples_per_segment
    end_idx = start_idx + samples_per_segment
    pressure_segment = pressure_list[start_idx:end_idx]
    heat_segment = heat_list[start_idx:end_idx]

    # Compute the start and end times for the segment
    start_time = (i * segment_duration)+4.6
    end_time = start_time + segment_duration

    # Compute the FFT for the segment
    pressure_fft = fft(pressure_segment)
    heat_fft = fft(heat_segment)

    # Compute the frequency bins
    freqs = fftfreq(samples_per_segment, d=1/fs)

    # Filter only the positive frequencies
    positive_freqs = freqs[:samples_per_segment // 2]
    pressure_fft_magnitude = np.abs(pressure_fft[:samples_per_segment // 2])
    heat_fft_magnitude = np.abs(heat_fft[:samples_per_segment // 2])

    # Apply the 500 Hz filter
    freq_filter = positive_freqs <= 500
    filtered_freqs = positive_freqs[freq_filter]
    filtered_pressure_fft = pressure_fft_magnitude[freq_filter]
    filtered_heat_fft = heat_fft_magnitude[freq_filter]
    filtered_heat_fft[filtered_freqs < 15] = 0

    # Normalize the FFT magnitudes
    normalized_pressure_fft = filtered_pressure_fft / np.max(filtered_pressure_fft)
    normalized_heat_fft = filtered_heat_fft / np.max(filtered_heat_fft)

    # Define the threshold and frequency error tolerance
    threshold = 0.7
    freq_tolerance = 1  # Hz

    # Find the indices where normalized_pressure_fft and normalized_heat_fft are greater than the threshold
    pressure_high_indices = np.where(normalized_pressure_fft > threshold)[0]
    heat_high_indices = np.where(normalized_heat_fft > threshold)[0]

    # Get the corresponding frequencies for these indices
    pressure_high_freqs = filtered_freqs[pressure_high_indices]
    heat_high_freqs = filtered_freqs[heat_high_indices]

    # Check for synchronization: frequencies close to each other within the tolerance
    syn_detected = []
    for p_freq in pressure_high_freqs:
        for h_freq in heat_high_freqs:
            if abs(p_freq - h_freq) <= freq_tolerance:
                syn_detected.append((p_freq, h_freq))

    # Print the results
    print("Synchronization Detected (Pressure Frequency, Heat Frequency):")
    for p_freq, h_freq in syn_detected:
        print(f"  Pressure: {p_freq:.2f} Hz, Heat: {h_freq:.2f} Hz")







    # Find the 10 highest peaks for pressure
    pressure_peak_indices = np.argsort(normalized_pressure_fft)[-10:][::-1]  # Indices of the 10 highest values
    pressure_peak_frequencies = filtered_freqs[pressure_peak_indices]
    pressure_peak_magnitudes = filtered_pressure_fft[pressure_peak_indices]

    # Find the 10 highest peaks for heat
    heat_peak_indices = np.argsort(normalized_heat_fft)[-10:][::-1]  # Indices of the 10 highest values
    heat_peak_frequencies = filtered_freqs[heat_peak_indices]
    heat_peak_magnitudes = filtered_heat_fft[heat_peak_indices]

    # Print the results for the segment
    print(f"Segment {i+1} ({start_time:.1f}s to {end_time:.1f}s):")
    print("Pressure Peaks:")
    for freq, mag in zip(pressure_peak_frequencies, pressure_peak_magnitudes):
        print(f"  Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
    print("Heat Peaks:")
    for freq, mag in zip(heat_peak_frequencies, heat_peak_magnitudes):
        print(f"  Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
    print()



    # Plot the FFT results for the segment
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Plot FFT of pressure
    axs[0].plot(filtered_freqs, filtered_pressure_fft, label=f'Pressure FFT (Segment {i+1})')
    axs[0].scatter(pressure_peak_frequencies, pressure_peak_magnitudes, color='red', label='Peaks')
    axs[0].set_title(f'FFT of Pressure (Segment {i+1}, {start_time:.1f}s to {end_time:.1f}s)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Magnitude')
    axs[0].legend()

    # Plot FFT of heat
    axs[1].plot(filtered_freqs, filtered_heat_fft, label=f'Heat FFT (Segment {i+1})', color='orange')
    axs[1].scatter(heat_peak_frequencies, heat_peak_magnitudes, color='red', label='Peaks')
    axs[1].set_title(f'FFT of Heat (Segment {i+1}, {start_time:.1f}s to {end_time:.1f}s)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Magnitude')
    axs[1].legend()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    # # Print the results for the segment
    # print(f"Segment {i+1} ({start_time:.1f}s to {end_time:.1f}s):")
    # print("Pressure Peaks:")
    # for freq, mag in zip(pressure_peak_frequencies, pressure_peak_magnitudes):
    #     print(f"  Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
    # print("Heat Peaks:")
    # for freq, mag in zip(heat_peak_frequencies, heat_peak_magnitudes):
    #     print(f"  Frequency: {freq:.2f} Hz, Magnitude: {mag:.2f}")
    # print()

    



    # # Compute the STFT of both signals
    # f, t, Zxx_pressure = stft(pressure_segment, fs, window='hann', nperseg=256, noverlap=128)
    # f, t, Zxx_heat = stft(heat_segment, fs, window='hann', nperseg=256, noverlap=128)

    # # Extract the phase of each signal
    # phase_pressure = np.angle(Zxx_pressure)  # Phase of pressure signal
    # phase_heat = np.angle(Zxx_heat)          # Phase of heat signal

    # # Compute the phase difference (in radians)
    # phase_difference = phase_heat - phase_pressure

    # # Wrap the phase difference to the range [-π, π]
    # phase_difference = np.angle(np.exp(1j * phase_difference))

    # # Plot the phase difference in the corresponding subplot
    # fig, axs = plt.subplots(1, 1, figsize=(10, 8))  # Create a new figure

    # im = axs.pcolormesh(t, f, phase_difference, shading='gouraud', cmap='hsv')
    # axs.set_ylabel('Frequency [Hz]')
    # axs.set_xlabel('Time [sec]')
    # axs.set_title(f'Phase Difference Between Pressure and Heat ({start_time:.1f}s to {end_time:.1f}s)')
    # axs.set_ylim(5, 500)

    # # Add a colorbar
    # cbar = fig.colorbar(im, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    # cbar.set_label('Phase Difference [rad]')

    # # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    # plt.close(fig)  # Explicitly close the figure


