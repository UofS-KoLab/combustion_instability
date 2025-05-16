import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
    
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

if __name__ == "__main__":
    args = parse_arguments()

    # Read the CSV file
    df = pd.read_csv(args.data_root)
    df.head()
    
    # Print the DataFrame
    # print(df)

    X = df.copy()
    df.drop(['filename','stability','case'],axis=1, inplace=True)
    #'flow_rate','hydrogen_ratio','heat_input'
    # Normalize the data (recommended for k-means)


    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)

    # # Step 3: Apply K-Means Clustering
    # # Initialize KMeans with 3 clusters
    # kmeans = KMeans(n_clusters=3, random_state=0)

    # # Fit the model to the scaled data
    # kmeans.fit(df_scaled)

    # # Get the cluster labels
    # labels = kmeans.labels_


    # Perform DBSCAN clustering
    # dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust eps and min_samples as needed
    # df['Cluster'] = dbscan.fit_predict(df_scaled)
    # gmm = GaussianMixture(n_components=3, random_state=42)
    # df['Cluster'] = gmm.fit_predict(df_scaled)
    # Perform spectral clustering

    #este va ganando
    spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors', random_state=42)
    df['Cluster'] = spectral.fit_predict(df_scaled)


    # Add the labels to the original DataFrame
    # df['Cluster'] = labels

    # Step 4: Analyze the results
    # Count the number of data points in each cluster
    print("Cluster distribution:\n", df['Cluster'].value_counts())

    # # Visualize the clusters (example with 2 features)
    # sns.scatterplot(x='peak1_freq_pressure', y='peak1_mag_pressure', hue='Cluster', data=df, palette='viridis')
    # plt.title('K-Means Clustering (k=3)')
    # plt.show()

    # features = df.columns
    # # Print the cluster centers
    # cluster_centers = kmeans.cluster_centers_
    # print("Cluster Centers:\n", cluster_centers)

    # # Calculate the importance of each feature
    # center_diffs = np.abs(cluster_centers - cluster_centers.mean(axis=0))
    # feature_importance = center_diffs.mean(axis=0)
    
    # for feature, importance in zip(features, feature_importance):
    #     print(f"Feature: {feature}, Importance: {importance}")


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    #pmt_highest_real_mag
    feature_1='rms'
    feature_2='sync_detected'
    feature_3='peak1_freq_pressure'

    scatter = ax.scatter(
        df[feature_1],
        df[feature_2], 
        df[feature_3], 
        # X['eq'],
        # X['rms_pmt'], 
        # X['rms'],
         
        c=df['Cluster'], 
        cmap='viridis', 
        marker='o'
    )
    
    # Create a legend with the cluster numbers
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)

    # Labels and title
    ax.set_title('3D K-means Clustering')
    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_zlabel(feature_3)
    # ax.set_xlabel('eq')
    # ax.set_ylabel('rms_pmt')
    # ax.set_zlabel('rms')
    
    plt.show()

    # Step 5: Save the results to a new CSV file
    # df.to_csv('clustered_data.csv', index=False)

    # # Optional: Determine the optimal number of clusters using the Elbow Method or Silhouette Score
    # from sklearn.metrics import silhouette_score

    # # Try different values of k and calculate the silhouette score
    # silhouette_scores = []
    # for k in range(2, 11):
    #     kmeans = KMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(df_scaled)
    #     score = silhouette_score(df_scaled, kmeans.labels_)
    #     silhouette_scores.append(score)

    # # Plot the silhouette scores
    # plt.plot(range(2, 11), silhouette_scores, marker='o')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('Silhouette Score')
    # plt.title('Silhouette Score for Optimal k')
    # plt.show()
    
