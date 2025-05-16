import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import argparse
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

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



# Generate sample data
def generate_data(n_samples=1000, n_features=2, n_components=3, random_state=42):
    """Generate synthetic data for clustering."""
    X, y = make_blobs(n_samples=n_samples, 
                     n_features=n_features, 
                     centers=n_components,
                     cluster_std=[1.0, 2.5, 0.5],
                     random_state=random_state)
    return X, y

# Fit GMM model
def fit_gmm(X, n_components=3, covariance_type='tied', random_state=42): #full
    """Fit Gaussian Mixture Model to data."""
    gmm = GaussianMixture(n_components=n_components,
                         covariance_type=covariance_type,
                         random_state=random_state)
    gmm.fit(X)
    return gmm

# Evaluate model
def evaluate_model(gmm, X):
    """Evaluate GMM model performance."""
    # Calculate BIC and AIC
    bic = gmm.bic(X)
    aic = gmm.aic(X)
    
    # Calculate silhouette score
    labels = gmm.predict(X)
    if len(np.unique(labels)) > 1:  # Silhouette score requires at least 2 clusters
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = None
    
    return {
        'BIC': bic,
        'AIC': aic,
        'Silhouette Score': silhouette,
        'Converged': gmm.converged_,
        'Iterations': gmm.n_iter_
    }

# Plot results
def plot_results(X, labels, means, covariances, title):
    """Plot clustering results (works for 'full', 'tied', 'diag', 'spherical')."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', alpha=0.6)
    
    # Handle different covariance types
    if covariances.ndim == 2:
        # 'tied' case: repeat the same covariance for all clusters
        covariances = np.tile(covariances, (len(means), 1, 1))
    
    for i, (mean, covar) in enumerate(zip(means, covariances)):
        # For 'diag' or 'spherical', reconstruct full covariance matrix
        if covar.ndim == 1:  # 'diag' case (diagonal elements only)
            covar = np.diag(covar)
        elif np.isscalar(covar):  # 'spherical' case (single variance value)
            covar = np.eye(len(mean)) * covar
        
        # Plot ellipse
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.degrees(np.arctan2(u[1], u[0]))
        
        ell = plt.matplotlib.patches.Ellipse(
            xy=mean,
            width=v[0],
            height=v[1],
            angle=angle,
            color=f'C{i}',
            alpha=0.5,
            fill=False
        )
        plt.gca().add_artist(ell)
        plt.scatter(mean[0], mean[1], marker='x', color=f'C{i}', s=100)
    
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.show()


if __name__ == "__main__":
    args = parse_arguments()
    df = pd.read_csv(args.data_root)
    df.head()



    X = df.copy() 
    X.drop(['filename','stability', 
            
    'rms_pressure','rms_pmt',
    'mean_pressure','mean_pmt',
    'variance_pressure', 'variance_pmt',
    'kurtosis_pressure', 'kurtosis_pmt',
    'skew_pressure', 
    'skew_pmt',
    'max_pressure', 
    'max_pmt',
    'min_pressure', 'min_pmt',

    # 'peak_to_peak_pressure', 

    'peak_to_peak_pmt',
    'rate_of_change_pressure', 'rate_of_change_pmt',


    # 'std_pmt',
    # 'std_pressure',
    



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
    
    "sync_total_peaks",
    "sync_frequency_1st_peak",
    # "score_instability",
    "ratio_amplitude_1"

    


            

            ],axis=1, inplace=True)
    X.head()
    X.info()



    # Create a list of the column names (head) of the DataFrame
    column_names = X.columns.tolist()

    # Print the list of column names
    print(column_names)

    X = X.to_numpy()
    
    # Split data (for evaluation purposes if needed)
    # X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    X_train=X
    print("le shape", X_train.shape)
    # Fit GMM model
    gmm = fit_gmm(X_train, n_components=4)

    # Evaluate model
    evaluation = evaluate_model(gmm, X_train)
    print("Model Evaluation:")
    for key, value in evaluation.items():
        print(f"{key}: {value}")
    
    # Predict clusters
    labels = gmm.predict(X_train)
    print(labels)
    probabilities = gmm.predict_proba(X_train)
    
    # Get model parameters
    means = gmm.means_
    covariances = gmm.covariances_
    
    # Plot results
    plot_results(X_train, labels, means, covariances, 
                "Gaussian Mixture Model Clustering")
    
    # Optional: Find optimal number of components using BIC/AIC
    n_components_range = range(1, 8)
    bic_scores = []
    aic_scores = []
    
    for n in n_components_range:
        gmm = fit_gmm(X_train, n_components=n)
        bic_scores.append(gmm.bic(X_train))
        aic_scores.append(gmm.aic(X_train))
    
    plt.figure(figsize=(10, 5))
    plt.plot(n_components_range, bic_scores, label='BIC')
    plt.plot(n_components_range, aic_scores, label='AIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Score')
    plt.title('BIC and AIC for Model Selection')
    plt.legend()
    plt.show()


    df['cluster_label'] = labels
    new_df = df[['filename','stability','cluster_label', 'std_pressure', 'std_pmt','norm_sync_score_first_5_peaks','score_instability','peak_to_peak_pressure']]
    new_df.to_csv(os.path.join(args.stability_file_to_save,f"{args.fuel_type}kmeans_thr_{args.threshold}_label.csv"), index=False)
 

    # Fit a classifier on GMM clusters
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X, labels)  # X = features, labels = GMM clusters

    # Get feature importances
    importance = clf.feature_importances_
    feature_names = column_names  # Exclude 'Cluster' column
    for i in range(len(importance)):
        importance[i] = round(importance[i], 5)
        print(f"Feature: {feature_names[i]}, Importance: {importance[i]}")
    # Plot
    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, importance)
    plt.title("Feature Importance for GMM Clusters (via Random Forest)")
    plt.xlabel("Importance Score")
    plt.show()