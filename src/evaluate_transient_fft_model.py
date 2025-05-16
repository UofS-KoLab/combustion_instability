import os
import argparse
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from tensorflow.keras.models import load_model
import shap
from tensorflow import keras

# Configure logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,  # Logs will be saved to this file
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--type_fuel", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument('--features', type=str, default="all", help='Comma-separated list of feature indices to select, or "all" to use all features.')
    
    return parser.parse_args()

def get_data(cache_file):
    """Load and preprocess data, or load from cache if available."""
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("There is not data available. Run the script called load_and_preprocess_transient.py to generate the data.")
    
    inputsALL, outputsALL_label = data
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    # label_mapping = {"Stable": 0, "Unstable": 1}
    label_mapping = {"Stable": 1, "Unstable": 0}
    outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)
    print(inputsALL.shape)
    print(outputsALL.shape)

    return inputsALL,outputsALL

def get_data_fuel(cache_file):
    """Load and preprocess data, or load from cache if available."""
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("There is not data available. Run the script called load_and_preprocess_transient.py to generate the data.")
    
    inputsALL, outputsALL_label,_ = data
    inputsALL=np.array(inputsALL)
    outputsALL_label=np.array(outputsALL_label)
    # Use the loaded data
    logging.info(f"Array 1 Shape:{inputsALL.shape}")
    logging.info(f"Array 2 Shape:{outputsALL_label.shape}")
    # Map string labels to integers
    label_mapping = {"Stable": 1, "Unstable": 0}
    # label_mapping = {"Stable": 0, "Unstable": 1}
    outputsALL = np.array([label_mapping[label] for label in outputsALL_label], dtype=np.int32)
    print(inputsALL.shape)
    print(outputsALL.shape)

    return inputsALL,outputsALL

def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach, "transient",f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    # Generate the cache file path
    # cache_file = os.path.join(folder_path, "data_s_to_u.pkl")
    cache_file = os.path.join(folder_path, "data_u_to_s.pkl")
    # cache_file = os.path.join(folder_path, "data.pkl")
    
    return cache_file

def get_cache_file_path_fuel(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    folder_path = os.path.join(project_root,"data", approach, "h2",f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    # Generate the cache file path
    cache_file = os.path.join(folder_path, "data.pkl")
    
    return cache_file

def get_log_file_path(project_root, window_size, approach):
    """Generate the log file path based on window size and approach."""
    # Create the logs folder if it doesn't exist
    logs_folder = os.path.join(project_root, "logs")
    os.makedirs(logs_folder, exist_ok=True)
    
    # Generate the log file name
    log_file = os.path.join(logs_folder, f"{approach}_transient_window_{window_size}ms.log")
    
    return log_file


if __name__ == "__main__":
    args = parse_arguments()

    # Generate the log file path and set up logging
    log_file = get_log_file_path(args.project_root, args.window_size, args.approach)
    setup_logging(log_file)
    
    # Generate the cache file path
    cache_file = get_cache_file_path(args.project_root, args.window_size, args.approach)
    
    # Load and preprocess data
    stability_label_dict=""
    transient_path=os.path.join(args.project_root,"data","raw","transient",args.type_fuel) #"hidrogen"
    filenames = [os.path.join(transient_path,"open_25kW_600slpm_70%to80%.csv")] #, os.path.join(transient_path,"open_25kW_600slpm_80%to70%.csv")
    inputsALL, outputsALL_label = get_data(cache_file)

    print(inputsALL.shape, outputsALL_label.shape)

    # cache_fuel= get_cache_file_path_fuel(args.project_root, args.window_size, args.approach)
    # input_fuel, output_fuel= get_data_fuel(cache_fuel)
    # # Log the shapes of the processed data
    logging.info(f"Processed data shapes - Inputs: {inputsALL.shape}, Outputs: {outputsALL_label.shape}")

    selected_features = list(map(int, args.features.split(',')))
    
    inputsALL = inputsALL[:, :, selected_features]

    model_path=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms",args.model_name,"lstm_model_fft.keras")
    model = load_model(model_path)

    predictions=[]
    # cache_prediction_file=os.path.join(args.project_root,"model",args.approach,f"window_{args.window_size}ms",args.model_name,"predictions_transient.pkl")
    # if os.path.exists(cache_prediction_file):
    #     logging.info("Loading data from cache...")
    #     predictions = load(cache_prediction_file)
    # else:
    print("New prediction")
    predictions = model.predict(inputsALL)
    # print(predictions)
    # predictions = (predictions > 0.5).astype(int) 
    print(predictions)
    # predictions = np.argmax(predictions)
    predictions = np.argmax(predictions, axis=1)
    print(predictions)
    print(len(predictions))
    # dump((predictions), cache_prediction_file)
    
    
    plt.figure(figsize=(10, 5))
    # predictions[predictions==0]=4
    # predictions[predictions==1]=5
    # predictions[predictions==2]=6
    # predictions[predictions==4]=1
    # predictions[predictions==5]=2
    # predictions[predictions==6]=0

    # print(predictions)
    # plt.scatter(time,outputsALL_label, label='Actual')
    plt.plot(predictions, label='Predictions', marker='x')

    for i in range(len(outputsALL_label)):
        if outputsALL_label[i] == 1:
            plt.axvspan(i - 0.5, i + 0.5, color='red', alpha=0.1)

    # plt.plot(outputsALL_label, label='Predictions', marker='x')
    # plt.yticks([0, 0.4], ["Stable", "Unstable"],fontsize=15)
    plt.xlabel('Time (s)',fontsize=15)
    # plt.ylabel('Prediction')
    # plt.title('Model Predictions')
    plt.grid(True)
    # plt.savefig(os.path.join(plot_path + f"/original_transient_{args.window_size}ms_{args.model_name}.png"), bbox_inches='tight', pad_inches=0.1)
    plt.show()

    
    accuracy = accuracy_score(outputsALL_label, predictions)
    
    # logging.info(f"Predictions: {predictions}")
    logging.info(f"Accuracy: {accuracy}")
    # print(f"Predictions: {predictions}")
    print(f"Accuracy: {accuracy}")

    # print(inputsALL.shape)
    # input_fuel = input_fuel[:, :, selected_features]
    # print(input_fuel.shape)
    # # model.summary()
    # X_train=input_fuel[:50]
    # X_test=inputsALL[50:60,:,:]
    # print(X_train.shape)
    # print(X_test.shape)
    
    # X_train_tf = tf.convert_to_tensor(X_train, dtype=tf.float32)
    # X_test_tf = tf.convert_to_tensor(X_test, dtype=tf.float32)
    
    # X_train_np = np.array(X_train, dtype=np.float32)
    # X_test_np = np.array(X_test, dtype=np.float32)

    # explainer = shap.GradientExplainer(model, inputsALL)
    # shap_values = explainer.shap_values(input_fuel)

    # explainer = shap.GradientExplainer(model, X_train_tf)

    # shap_values = explainer.shap_values(X_test_tf)
    # shap_values = np.array(shap_values)
    # print(shap_values.shape)

    # shap.summary_plot(shap_values, input_fuel[:10])
    # shap.force_plot(explainer.expected_value[0], shap_values[0], input_fuel[100:150])



    # X_train_tf = tf.convert_to_tensor(input_fuel[:50], dtype=tf.float32)
    # explainer = shap.DeepExplainer(model, X_train_tf)

    # # Explain a few test instances
    # shap_values = explainer.shap_values(inputsALL[100:110,:,:])

    # # Convert to NumPy array for visualization if necessary
    # shap_values = np.array(shap_values)



    # cm = confusion_matrix(outputsALL_label, predictions)
    # logging.info(f"Confusion Matrix:\n{cm}")
    # print(f"Confusion Matrix:\n{cm}")
    
    # # Plot confusion matrix
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Stable', 'Unstable'])
    # disp.plot(cmap='Blues')

    # # Specify the folder and filename
    # plot_path = os.path.join(args.project_root, "plot", args.approach)

    # # Create the folder if it doesn't exist
    # if not os.path.exists(plot_path):
    #     os.makedirs(plot_path)

    # # Save the plot
    # plt.savefig(plot_path + f"/confusion_matrix_transient_{args.window_size}ms_{args.model_name}.png", bbox_inches='tight', pad_inches=0.1)
    # plt.show()

    # # Number of points
    # num_points = 160#160#480#1600

    # # Time vector (0 to 48 seconds, with 480 points)
    # time = np.linspace(0, 48, num_points)
    # print(len(time))
    # plt.figure(figsize=(10, 5))
    # # plt.scatter(time,outputsALL_label, label='Actual')
    # plt.scatter(time,predictions*0.4, label='Predictions', marker='x')
    # plt.yticks([0, 0.4], ["Stable", "Unstable"],fontsize=15)
    # plt.xlabel('Time (s)',fontsize=15)
    # # plt.ylabel('Prediction')
    # # plt.title('Model Predictions')
    # plt.grid(True)
    # plt.savefig(os.path.join(plot_path + f"/prediction_transient_{args.window_size}ms_{args.model_name}.png"), bbox_inches='tight', pad_inches=0.1)
    # plt.show()

    # # #depends on the window size
    # # original_transient=[0]*193
    # # original_transient.extend([0.4]*103)
    # # original_transient.extend([0]*184)
    # # plt.figure(figsize=(10, 5))
    # # # plt.scatter(time,outputsALL_label, label='Actual')
    # # plt.scatter(time,original_transient, label='Predictions', marker='x')
    # # plt.yticks([0, 0.4], ["Stable", "Unstable"],fontsize=15)
    # # plt.xlabel('Time (s)',fontsize=15)
    # # # plt.ylabel('Prediction')
    # # # plt.title('Model Predictions')
    # # plt.grid(True)
    # # plt.savefig(os.path.join(plot_path + f"/original_transient_{args.window_size}ms_{args.model_name}.png"), bbox_inches='tight', pad_inches=0.1)
    # # plt.show()
