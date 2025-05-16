import pandas as pd
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--old_data_file", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--cluster_label_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument("--fuel_type", type=str, required=True, help="Type of fuel used in the experiment.")
    parser.add_argument("--threshold", type=str, required=True, help="Type of fuel used in the experiment.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Read the CSV file
    df_data = pd.read_csv(args.data_root)
    df_cluster_labels = pd.read_csv(args.cluster_label_file)

    # Merge df_data_sorted with df_cluster_labels based on the filename column
    df_merged = pd.merge(df_data, df_cluster_labels[['filename', 'cluster_label']], on='filename', how='left')


    # df_merged['freq_rate'] = df_merged['peak1_freq_pressure']/df_merged['peak1_freq_pmt']

    feature_choosen="std_pressure"

    max_peak1_freq_pressure = df_merged[feature_choosen].max()
    df_merged['stability_numeric'] = df_merged['stability'].map({'Stable': 0, 'Unstable': 2})
    
    df_data_sorted = df_merged.sort_values(by=feature_choosen, ascending=True)
    # Print the sorted DataFrame
    print(df_merged)

    # Plot peak1_freq_pressure with samples on x-axis and color by cluster_label
    plt.figure(figsize=(15, 6))
    scatter = plt.scatter(df_data_sorted['filename'], df_data_sorted[feature_choosen], c=df_data_sorted['cluster_label'], cmap='viridis', marker='o')
    
    # Add color bar
    plt.colorbar(scatter, label='Cluster Label')
    # Plot stability_numeric on the same plot
    
    plt.xticks(rotation=90)
    
    # Add labels and title
    plt.xlabel('Samples')
    plt.ylabel(feature_choosen)
    plt.title(f'{feature_choosen} by Samples and Cluster Label thr:{args.threshold}')
    
    
    
    plt.tight_layout()
    # Show the plot
    # plt.show()
    # plt.savefig(os.path.join(args.project_root,"plot","clusters","no_sync_total_peaks",f"thr_{args.threshold}_by_{feature_choosen}s.png"), bbox_inches='tight')
    # plt.savefig(os.path.join(args.project_root,"plot","clusters","all_features",f"thr_{args.threshold}_by_{feature_choosen}s.png"), bbox_inches='tight')
    # plt.savefig(os.path.join(args.project_root,"plot","clusters","no_peak1",f"thr_{args.threshold}_by_{feature_choosen}s.png"), bbox_inches='tight')
    plt.savefig(os.path.join(args.project_root,"plot","clusters",f"{args.fuel_type}_thr_{args.threshold}_by_{feature_choosen}s.png"), bbox_inches='tight')
    





    df_data_sorted = df_merged.sort_values(by='std_pressure', ascending=True)
     # Plot peak1_freq_pressure with samples on x-axis and color by cluster_label
    plt.figure(figsize=(15, 6))

    ax = plt.gca()

    # Highlight background where stability_numeric is 2
    for i, val in enumerate(df_data_sorted['stability_numeric']):
        if val == 2:
            ax.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.3)

    # Plot cluster_label with different colors
    scatter = plt.scatter(df_data_sorted['filename'], df_data_sorted['cluster_label'], c=df_data_sorted['cluster_label'], cmap='viridis', marker='o')

    # Add color bar
    plt.colorbar(scatter, label='Cluster Label')
    # scatter = plt.scatter(df_data_sorted['filename'], df_data_sorted['cluster_label'], c=df_merged['cluster_label'], cmap='viridis', marker='o')
    # plt.scatter(df_data_sorted['filename'], df_data_sorted['cluster_label'])
    # plt.scatter(df_data_sorted['filename'], df_data_sorted['stability_numeric'], label=f'Stability (0=Stable, 2=Unstable)', color='red')

    # Add color bar
    # plt.colorbar(scatter, label='Cluster Label')
    # Plot stability_numeric on the same plot
    
    plt.xticks(rotation=90)
    
    # Add labels and title
    plt.xlabel('Samples')
    plt.ylabel('cluster_label')
    plt.title(f'Samples order by std_pressure') #thr:{args.threshold}
    
    
    
    plt.tight_layout()
    # Show the plot
    # plt.show()
    # plt.savefig(os.path.join(args.project_root,"plot","clusters","no_sync_total_peaks",f"thr_{args.threshold}_old_clusters.png"), bbox_inches='tight')
    #plt.savefig(os.path.join(args.project_root,"plot","clusters","all_features",f"thr_{args.threshold}_old_clusters.png"), bbox_inches='tight')
    # plt.savefig(os.path.join(args.project_root,"plot","clusters","no_peak1",f"thr_{args.threshold}_old_clusters.png"), bbox_inches='tight')
    plt.savefig(os.path.join(args.project_root,"plot","clusters",f"{args.fuel_type}_thr_{args.threshold}_old_clusters.png"), bbox_inches='tight')
    

    # Calculate the percentage of data points with stability=2 that have cluster_label=1
    # Calculate the percentage of data points with stability=2 that have cluster_label=1
    stability_2 = df_merged[df_merged['stability_numeric'] == 2]
    total_stability_2 = len(stability_2)
    cluster_label_0_in_stability_2 = len(stability_2[stability_2['cluster_label'] == 0])
    cluster_label_1_in_stability_2 = len(stability_2[stability_2['cluster_label'] == 1])
    cluster_label_2_in_stability_2 = len(stability_2[stability_2['cluster_label'] == 2])

    percentage_0 = (cluster_label_0_in_stability_2 / total_stability_2) * 100 if total_stability_2 > 0 else 0
    percentage_1 = (cluster_label_1_in_stability_2 / total_stability_2) * 100 if total_stability_2 > 0 else 0
    percentage_2 = (cluster_label_2_in_stability_2 / total_stability_2) * 100 if total_stability_2 > 0 else 0

    stability_0 = df_merged[df_merged['stability_numeric'] == 0]
    total_stability_0 = len(stability_0)
    cluster_label_0_in_stability_0 = len(stability_0[stability_0['cluster_label'] == 0])
    cluster_label_1_in_stability_0 = len(stability_0[stability_0['cluster_label'] == 1])
    cluster_label_2_in_stability_0 = len(stability_0[stability_0['cluster_label'] == 2])

    percentage_0_stable = (cluster_label_0_in_stability_0 / total_stability_0) * 100 if total_stability_0 > 0 else 0
    percentage_1_stable = (cluster_label_1_in_stability_0 / total_stability_0) * 100 if total_stability_0 > 0 else 0
    percentage_2_stable = (cluster_label_2_in_stability_0 / total_stability_0) * 100 if total_stability_0 > 0 else 0

    # Save the results to a text file
    # output_txt_path = os.path.join(args.project_root, "plot", "clusters","no_sync_total_peaks", f"thr_{args.threshold}_percentages.txt")
    # output_txt_path = os.path.join(args.project_root, "plot", "clusters","all_features", f"thr_{args.threshold}_percentages.txt")
    # output_txt_path = os.path.join(args.project_root, "plot", "clusters","no_peak1", f"thr_{args.threshold}_percentages.txt")
    output_txt_path = os.path.join(args.project_root, "plot", "clusters", f"{args.fuel_type}_thr_{args.threshold}_percentages.txt")
    

    with open(output_txt_path, 'w') as f:
        f.write(f"Results for threshold {args.threshold}:\n")
        f.write(f"total unstable: {total_stability_2}, total cluster 0: {cluster_label_0_in_stability_2}, percentage: {percentage_0:.2f}%\n")
        f.write(f"total unstable: {total_stability_2}, total cluster 1: {cluster_label_1_in_stability_2}, percentage: {percentage_1:.2f}%\n")
        f.write(f"total unstable: {total_stability_2}, total cluster 2: {cluster_label_2_in_stability_2}, percentage: {percentage_2:.2f}%\n")
        f.write(f"total stable: {total_stability_0}, total cluster 0: {cluster_label_0_in_stability_0}, percentage: {percentage_0_stable:.2f}%\n")
        f.write(f"total stable: {total_stability_0}, total cluster 1: {cluster_label_1_in_stability_0}, percentage: {percentage_1_stable:.2f}%\n")
        f.write(f"total stable: {total_stability_0}, total cluster 2: {cluster_label_2_in_stability_0}, percentage: {percentage_2_stable:.2f}%\n")

    print(f"Results saved to {output_txt_path}")