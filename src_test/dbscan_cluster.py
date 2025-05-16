import pandas as pd
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--old_data_file", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--stability_file_to_save", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument("--fuel_type", type=str, required=True, help="Type of fuel used in the experiment.")
    parser.add_argument("--threshold", type=str, required=True, help="Type of fuel used in the experiment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Read the CSV file
    df = pd.read_csv(args.data_root)
    df.head()
    X = df.copy()
    
    # freq_pmt = X['peak1_freq_pmt']
    # freq_pressure = X['peak1_freq_pressure']

    # Create a new column 'ratio_frequency' with the division of 'freq_pmt' by 'freq_pressure'
    # X['ratio_frequency'] = freq_pmt / freq_pressure

    # X = pd.get_dummies(df, columns=['case'], prefix='state')
    
    # Drop non-numeric and unnecessary columns
    # X.drop(['filename', 'stability', 'case', 'flow_rate', 'hydrogen_ratio', 'heat_input',
    #         'sync_mag_pressure','sync_mag_pmt'], axis=1, inplace=True)
    X.drop(['filename', 'stability',
            'mean_pressure','mean_pmt',
             
              'variance_pressure', 'variance_pmt',
              
                # 'kurtosis_pressure', "kurtosis_pmt",
                 
                #  'min_pmt',
                
                #  'rate_of_change_pressure', 'rate_of_change_pmt',
                  'rms_pressure','rms_pmt',
                   
                #    "skew_pressure","skew_pmt",
                #    "peak1_freq_pressure", "peak1_mag_pressure", "peak1_freq_pmt", "peak1_mag_pmt",
                #    "peak2_freq_pressure", "peak2_mag_pressure", "peak2_freq_pmt", "peak2_mag_pmt",
                #    "sync_detected_range_1","sync_detected_range_2",
                #    'freq_pressure_peak_range_1','peak_mag_pressure_range_1','peak_mag_p_pmt_range_1',
                #    'freq_pressure_peak_range_2','peak_mag_pressure_range_2','peak_mag_p_pmt_range_2'
                'phase_peak1_pressure','phase_peak2_pressure','phase_peak1_pmt','phase_peak2_pmt',
                'sync_detected_range_1','sync_detected_range_2',
                # ,'freq_pressure_peak_range_1','peak_mag_pressure_range_1','peak_mag_p_pmt_range_1',
                # 'sync_detected_range_2','freq_pressure_peak_range_2','peak_mag_pressure_range_2','peak_mag_p_pmt_range_2',
                # 'norm_sync_score_range1_peaks','norm_sync_score_range2_peaks',
                # 'phase_1_differece','phase_2_differece'
                'norm_sync_score_first_10_peaks'
            
            
            
            ], axis=1, inplace=True)
    # Normalize data
    cols = X.columns
    ms = MinMaxScaler()
    X_scaled = ms.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=cols)
    
    feature_variance = X_scaled.var()
    print("Feature Variance:\n", feature_variance)
    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.05,min_samples=20)  # Tune eps and min_samples for optimal results
    labels = dbscan.fit_predict(X_scaled)
    
    df['cluster_label'] = labels
    
    # Save the DataFrame to a CSV file
    new_df = df[['filename', 'stability', 'cluster_label']]
    new_df.to_csv(os.path.join(args.stability_file_to_save, f"{args.fuel_type}_dbscan_thr_{args.threshold}_label.csv"), index=False)
    
    unique_labels = set(labels)
    print(f"Unique cluster labels: {unique_labels}")
    
    # 3D Visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    feature_1 = 'std_pressure'
    feature_2 = 'skew_pressure'
    feature_3 = 'kurtosis_pressure'
    
    scatter = ax.scatter(
        X_scaled[feature_1],
        X_scaled[feature_2],
        X_scaled[feature_3],
        c=df['cluster_label'],
        cmap='viridis',
        marker='o'
    )
    
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    
    ax.set_title(f"3D DBSCAN Clustering {args.threshold}")
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
    
    plt.show()
