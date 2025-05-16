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
    parser.add_argument("--stability_file_to_save", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument("--fuel_type", type=str, required=True, help="Type of fuel used in the experiment.")
    parser.add_argument("--threshold", type=str, required=True, help="Type of fuel used in the experiment.")
    return parser.parse_args()


def create_combined_dataframe(list_names_files,folder_path):
    """
    Create a combined DataFrame from all CSV files in the specified folder.

    :param folder_path: Path to the folder.
    :return: Combined DataFrame.
    """
    csv_files = list_names_files
    df_list = []

    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)
    return combined_df

if __name__ == "__main__":
    args = parse_arguments()

    # fuel_types_names = ["h2","A-2","C-1","C-5","C-9","Jet-A","JP8_HRJ"]
    fuel_types_names = ["transient"]
    list_names_files=[]
    for fuel in fuel_types_names:
        list_names_files.append(f"{fuel}_cluster_thr_{args.threshold}.csv")
    
    folder_path=os.path.join(args.project_root,"data","cluster")
    All_fuels_X=create_combined_dataframe(list_names_files,folder_path)
    print(All_fuels_X)
    All_fuels_X['ratio_freq_1'] = All_fuels_X['freq_1_pressure'] / All_fuels_X['freq_1_pmt']
    All_fuels_X['ratio_freq_2'] = All_fuels_X['freq_2_pressure'] / All_fuels_X['freq_2_pmt']
    All_fuels_X['ratio_freq_3'] = All_fuels_X['freq_3_pressure'] / All_fuels_X['freq_3_pmt']

    # Read the CSV file
    df = pd.read_csv(args.data_root)
    # df = df.drop_duplicates(subset='filename', keep='first')
    # df=All_fuels_X
    df.head()
    X = df.copy() 
    # X = X.drop_duplicates(subset='filename', keep='first')

     
    
    # freq_pressure_2 = X['peak2_freq_pressure']
    # freq_pressure = X['peak1_freq_pressure']

    # Create a new column 'ratio_frequency' with the division of 'freq_pmt' by 'freq_pressure'
    # X['ratio_freq_1'] = X['freq_1_pressure'] / X['freq_1_pmt']
    # X['ratio_freq_2'] = X['freq_2_pressure'] / X['freq_2_pmt']
    # X['ratio_freq_3'] = X['freq_3_pressure'] / X['freq_3_pmt']
    # X['avergae_phase_diff']= (X['phase_1_differece']+X['phase_2_differece'])/2
    # print(X['filename'])
    # print(X['ratio_freq_1'])

    # print(X['freq_1_pressure'])
    # print(X['freq_1_pmt'])
    # X.info()

    le = LabelEncoder()
    le_1 = LabelEncoder()
    le_2 = LabelEncoder()

    #

    # X.drop(['filename','stability',
    #         "peak1_freq_pressure", "peak1_mag_pressure", "peak1_freq_pmt", "peak1_mag_pmt", 
    #         "peak2_freq_pressure", "peak2_mag_pressure", "peak2_freq_pmt", "peak2_mag_pmt", 
    #         "sync_detected", "sync_frequency", "sync_mag_pressure", "sync_mag_pmt", "sync_total_peaks", 
    #         "total_peaks_pressure", "total_peaks_pmt", 
    #         "total_energy_p", "total_energy_pmt", 
    #         "peak_freq_5_100_p", "peak_mag_5_100_p", "peak_freq_5_100_pmt", "peak_mag_5_100_pmt", 
    #         "peak_freq_100_200_p", "peak_mag_100_200_p", "peak_freq_100_200_pmt", "peak_mag_100_200_pmt", 
    #         "peak_freq_200_300_p", "peak_mag_200_300_p", "peak_freq_200_300_pmt", "peak_mag_200_300_pmt", 
    #         "peak_freq_300_400_p", "peak_mag_300_400_p", "peak_freq_300_400_pmt", "peak_mag_300_400_pmt", 
    #         "peak_freq_400_500_p", "peak_mag_400_500_p", "peak_freq_400_500_pmt", "peak_mag_400_500_pmt"

    #         ],axis=1, inplace=True)

    # X.drop(['filename','stability','mean_pressure','mean_pmt',
    #          'std_pressure', 'std_pmt',
    #           'variance_pressure', 'variance_pmt',
    #           'skew_pressure', 'skew_pmt',
    #             'kurtosis_pressure', 'kurtosis_pmt',
    #             'max_pressure', 'max_pmt',
    #             'min_pressure', 'min_pmt',
    #              'peak_to_peak_pressure', 'peak_to_peak_pmt',
    #              'rate_of_change_pressure', 'rate_of_change_pmt',
    #               'rms_pressure','rms_pmt'

    #         ],axis=1, inplace=True)

    # "filename": name,
    #     "stability": state,
    #     "peak1_freq_pressure": peak1_freq_pressure,
    #     "peak1_mag_pressure": peak1_mag_pressure,
    #     "peak1_freq_pmt": peak1_freq_pmt,
    #     "peak1_mag_pmt": peak1_mag_pmt,
    #     "peak2_freq_pressure": peak2_freq_pressure,
    #     "peak2_mag_pressure": peak2_mag_pressure,
    #     "peak2_freq_pmt": peak2_freq_pmt,
    #     "peak2_mag_pmt": peak2_mag_pmt,
    #     "sync_detected_range_1": sync_detected_range_1,
    #     "freq_pressure_peak_range_1": freq_pressure_peak_range_1,
    #     "peak_mag_pressure_range_1": peak_mag_pressure_range_1,
    #     "peak_mag_p_pmt_range_1": peak_mag_p_pmt_range_1,
    #     "sync_detected_range_2": sync_detected_range_2,
    #     "freq_pressure_peak_range_2": freq_pressure_peak_range_2,
    #     "peak_mag_pressure_range_2": peak_mag_pressure_range_2,
    #     "peak_mag_p_pmt_range_2": peak_mag_p_pmt_range_2,

    # 'mean_pressure','mean_pmt',
    #         'variance_pressure', 'variance_pmt',
            
                
    #             'rate_of_change_pressure', 'rate_of_change_pmt',
    #             'rms_pressure','rms_pmt',
                   
    #             "peak1_freq_pressure", "peak1_mag_pressure", "peak1_freq_pmt", "peak1_mag_pmt",
    #             "peak2_freq_pressure", "peak2_mag_pressure", 
    #             "peak2_freq_pmt", 
    #             "peak2_mag_pmt",
    #             'freq_pressure_peak_range_1','peak_mag_pressure_range_1','peak_mag_p_pmt_range_1',
    #             'phase_peak1_pressure','phase_peak2_pressure','phase_peak1_pmt','phase_peak2_pmt',
    #             'sync_detected','sync_total_peaks',
    #             'norm_sync_score_first_5_peaks',
    #             'sync_detected_range_2',
    #             'peak_mag_pressure_range_2',
    #             'peak_mag_p_pmt_range_2',
    #             'freq_pressure_peak_range_1',
    #             'kurtosis_pmt',
    #             'norm_sync_score_range2_peaks',
    #             'min_pmt',
    #             'freq_pressure_peak_range_2',
    #             'peak_to_peak_pmt',
    #             'max_pmt',
    #             'std_pmt',
    #             'peak_mag_p_pmt_range_1',
    #             'min_pressure'

   
    X.drop(['filename','stability', 
            
    'rms_pressure','rms_pmt',
    'mean_pressure','mean_pmt',
    'variance_pressure', 'variance_pmt',
    'kurtosis_pressure', 'kurtosis_pmt',
    'skew_pressure', 'skew_pmt',
    'max_pressure', 'max_pmt',
    'min_pressure', 'min_pmt',
    # 'peak_to_peak_pressure', 
    'peak_to_peak_pmt',
    'rate_of_change_pressure', 'rate_of_change_pmt',
    # 'std_pmt',
    



    'ratio_freq_1',
    'ratio_freq_2',
    'ratio_freq_3',
            
    "peak1_freq_pressure",
    "peak1_mag_pressure",
    "peak1_freq_pmt",
    "peak1_mag_pmt",
    "peak2_freq_pressure",
    "peak2_mag_pressure",
    "peak2_freq_pmt",
    "peak2_mag_pmt",
    
    "sync_detected_range_1",
    "freq_pressure_peak_range_1",
    "peak_mag_pressure_range_1",
    "peak_mag_p_pmt_range_1",
    "sync_detected_range_2",
    "freq_pressure_peak_range_2",
    "peak_mag_pressure_range_2",
    "peak_mag_p_pmt_range_2",
    
    "phase_1_differece",
    "phase_2_differece",
    "phase_peak1_pressure",
    "phase_peak2_pressure",
    "phase_peak1_pmt",
    "phase_peak2_pmt",
    
    "norm_sync_score_first_10_peaks",
    # "norm_sync_score_first_5_peaks",
    "norm_sync_score_range1_peaks",
    "norm_sync_score_range2_peaks",
    
    "highest_mag_p_rg1",
    "highest_mag_pmt_rg1",
    "highest_freq_p_rg1",
    "highest_freq_pmt_rg1",
    "highest_sync_rg1",
    "phase_differece_rg1",
    
    "freq_1_pressure",
    "mag_1_pressure",
    "freq_1_pmt",
    "mag_1_pmt",
    "phase_1_difference",
    "freq_2_pressure",
    "mag_2_pressure",
    "freq_2_pmt",
    "mag_2_pmt",
    "phase_2_difference",
    "freq_3_pressure",
    "mag_3_pressure",
    "freq_3_pmt",
    "mag_3_pmt",
    "phase_3_difference",
    "freq_4_pressure",
    "mag_4_pressure",
    "freq_4_pmt",
    "mag_4_pmt",
    "phase_4_difference",
    "freq_5_pressure",
    "mag_5_pressure",
    "freq_5_pmt",
    "mag_5_pmt",
    "phase_5_difference",
    
    "sync_detected",
    "sync_frequency_1st_peak",
    
    "sync_total_peaks",

    


            

            ],axis=1, inplace=True)
    X.head()
    X.info()
    num_clusters=3
    cols = X.columns
    ms = MinMaxScaler()
    X = ms.fit_transform(X)
    X = pd.DataFrame(X, columns=[cols])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0) 
    kmeans.fit(X)
    # clusters = kmeans.fit_predict(X)
    # X['cluster_label'] = clusters
    # new_df = X[['cluster_label', 'std_pressure','ratio_freq_1','ratio_freq_2']]
    # new_df.to_csv(os.path.join(args.stability_file_to_save,f"{args.fuel_type}new_{args.threshold}_label.csv"), index=False)
    
    
    


    # print(X)
    kmeans.cluster_centers_
    # print(kmeans.inertia_)
    labels = kmeans.labels_
    # print(labels)
    features = X.columns
    # Print the cluster centers
    cluster_centers = kmeans.cluster_centers_
    print("Cluster Centers:\n", cluster_centers)

    # Calculate the importance of each feature
    center_diffs = np.abs(cluster_centers - cluster_centers.mean(axis=0))
    feature_importance = center_diffs.mean(axis=0)
    
    temporal_feature=[]
    temporal_importance=[]
    for feature, importance in zip(features, feature_importance):
        print(f"Feature: {feature}, Importance: {importance}")
        temporal_feature.append(str(feature[0]))
        temporal_importance.append(importance)
    
    paired_feature_importance = list(zip(temporal_feature, temporal_importance))
    paired_sorted_f_i=sorted(paired_feature_importance, key=lambda x: x[1], reverse=True)
    sorted_feature, sorted_importance = zip(*paired_sorted_f_i)
    sorted_feature=list(sorted_feature)
    # print("Sorted Feature:", sorted_feature)
    # print("Sorted Importance:", sorted_importance)
    print("\n")
    for i in range(len(sorted_feature)):
        print(f"{sorted_feature[i]},{sorted_importance[i]}")
    print("\n")
    # Add the labels to the DataFrame
    df['cluster_label'] = labels
    # df['score_10']= X['norm_sync_score_first_10_peaks']
    # df['ratio_1']=X['ratio_freq_1']
    # df['ratio_2']=X['ratio_freq_2']
    # X['cluster_label'] = labels
    
    # Save the DataFrame to a CSV file
    new_df = df[['filename','stability','cluster_label', 'std_pressure', 'std_pmt','peak_to_peak_pressure','score_instability','norm_sync_score_first_5_peaks','norm_sync_score_first_10_peaks','sync_frequency_1st_peak','sync_detected','sync_total_peaks']]
    new_df.to_csv(os.path.join(args.stability_file_to_save,f"{args.fuel_type}kmeans_thr_{args.threshold}_label.csv"), index=False)
    # new_df.to_csv(os.path.join(args.stability_file_to_save,f"allkmeans_thr_{args.threshold}_label.csv"), index=False)
    


    # Filter the DataFrame where cluster_label is 1
    filtered_df = new_df[new_df['cluster_label'] == 1]

    # Get the set of unique filenames
    unique_filenames = set(filtered_df['filename'].unique())

    # Print the set of unique filenames
    # print(len(unique_filenames))
    # print(unique_filenames)

    unique_filenames_0 = df[df['cluster_label'] == 0]['filename'].unique()
    unique_filenames_1 = df[df['cluster_label'] == 1]['filename'].unique()
    unique_filenames_2 = df[df['cluster_label'] == 2]['filename'].unique()
    print("Unique filenames where cluster_label is 0:", len(unique_filenames_0))
    print("Unique filenames where cluster_label is 1:", len(unique_filenames_1))
    print("Unique filenames where cluster_label is 2:", len(unique_filenames_2))
    # print("Unique filenames where cluster_label is 0:")
    # print(unique_filenames)
    # print(len(unique_filenames))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    #pmt_highest_real_mag
 
    feature_1=sorted_feature[2]
    feature_2=sorted_feature[1]
    feature_3=sorted_feature[0]

    scatter = ax.scatter(
        X[feature_1],
        X[feature_2], 
        X[feature_3], 
         
        c=df['cluster_label'], 
        cmap='viridis', 
        marker='o'
    )
    
    # Create a legend with the cluster numbers
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    # Labels and title
    ax.set_title(f"3D K-means Clustering ") #{args.threshold}
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
    
    plt.show()












#     #to test
#     top_names=['sync_detected','phase_1_difference','sync_total_peaks','phase_2_difference'
# ,'skew_pmt'
# ,'phase_3_difference'
# ,'kurtosis_pressure'
# ,'skew_pressure'
# ,'std_pressure'
# ,'norm_sync_score_first_10_peaks'
# ,'norm_sync_score_first_5_peaks'
# ,'ratio_freq_2'
# ,'max_pressure'
# ,'peak_to_peak_pressure'
# ,'ratio_freq_3'
# ,'min_pressure'
# ,'std_pmt'
# ,'ratio_freq_1'
# ,'max_pmt'
# ,'min_pmt']
#     test_top=All_fuels_X[top_names]

#     transient_new=All_fuels_X.copy()
       
#     # Predict the clusters for the new data
#     new_clusters = kmeans.predict(test_top)

#     # Add the predicted clusters to the new data (optional)
#     test_top['Cluster'] = new_clusters
#     transient_new['Cluster'] = new_clusters


#     # Save the DataFrame to a CSV file
#     output_csv_path = 'test_top_with_clusters.csv'  # Replace with the desired output path
#     transient_new.to_csv(output_csv_path, index=False)
#     print(f"DataFrame saved to {output_csv_path}")




    top_features = sorted_feature[:5]

    # Create a new feature matrix with top features
    X_top = X[top_features]

    # Refine clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_top)

    # Evaluate clustering
    from sklearn.metrics import silhouette_score
    score = silhouette_score(X_top, clusters)
    print(f"Silhouette Score: {score}")

