import os
import argparse
import json
import numpy as np
import pandas as pd
from scipy import signal
from joblib import dump, load
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

import shap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Activation, Conv2D, Flatten, Dropout, BatchNormalization, Add, Concatenate, MaxPooling1D, AveragePooling1D #, LocallyConnected2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.legacy import Adam, RMSprop #for windows
#from tensorflow.keras.optimizers import Adam, RMSprop #for linux
from tensorflow.keras import mixed_precision

#For windows
# Check if TensorFlow detects a GPU
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPU is available!")
    print("Details of GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected. TensorFlow is using the CPU.")
# Enable multi-threading
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)

#For linux 
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.set_logical_device_configuration(
#                 gpu,
#                 [tf.config.LogicalDeviceConfiguration(memory_limit=1024 * 8)]  # Adjust memory limit if needed
#             )
#         logical_gpus = tf.config.list_logical_devices('GPU')
#         print(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs configured.")
#     except RuntimeError as e:
#         print(f"GPU Configuration Error: {e}")
# else:
#     print("No GPUs found, using CPU instead.")
    

# Configure logging
def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,  # Logs will be saved to this file
        level=logging.INFO,  # Log level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

# Disable unnecessary logging for windows
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Process combustion instability data.")
    parser.add_argument("--data_root", type=str, required=True, help="Path to the main data directory.")
    parser.add_argument("--project_root", type=str, required=True, help="Path to the main directory.")
    parser.add_argument("--stability_file", type=str, required=True, help="Path to the stability labeling CSV file.")
    parser.add_argument("--duration_sample_ms", type=int, required=True, help="Total duration of the sample of the time series in ms.")
    parser.add_argument("--window_size", type=int, required=True, help="Size of the time series window in ms (e.g., 100 for 100ms).")
    parser.add_argument("--approach", type=str, required=True, choices=["time_series", "fft"], help="Approach: 'time_series' or 'fft'.")
    parser.add_argument('--lstm_units1', type=int, default=160, help="Number of units in the first LSTM layer.")
    parser.add_argument('--lstm_units2', type=int, default=192, help="Number of units in the second LSTM layer.")
    parser.add_argument('--dropout_rate1', type=float, default=0.3, help="Dropout rate for the first LSTM layer.")
    parser.add_argument('--dropout_rate2', type=float, default=0.0, help="Dropout rate for the second LSTM layer.")
    parser.add_argument('--l2_regularizer1', type=float, default=0.0099, help="L2 regularization for the first LSTM layer.")
    parser.add_argument('--l2_regularizer2', type=float, default=0.0004, help="L2 regularization for the second LSTM layer.")
    parser.add_argument('--learning_rate', type=float, default=0.008, help="Learning rate for the optimizer.")
    parser.add_argument('--epochs', type=int, default=150, help="Number of epochs to train the model.")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument('--patience', type=int, default=3, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the test split.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument('--features', type=str, default="all", help='Comma-separated list of feature indices to select, or "all" to use all features.')
    
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


def get_filenames(data_root):
    """Get all Excel files in the specified directory."""
    return [os.path.join(data_root, f) for f in os.listdir(data_root) if f.endswith(".csv")]

def get_cache_file_path(project_root, window_size, approach):
    """Generate the cache file path based on window size and approach."""
    # Create the folder path
    fuel_type="h2"
    folder_path = os.path.join(project_root,"data", approach,fuel_type, f"window_{window_size}ms")
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    
    # Generate the cache file path
    cache_file = os.path.join(folder_path, "data.pkl")

    print(os.path.join(folder_path, "data.pkl"))

    # cache_file = os.path.join(folder_path, "data_cluster_3.pkl")

    # print(os.path.join(folder_path, "data_cluster_3.pkl"))
    
    return cache_file

def get_data(cache_file, filenames, stability_label_dict):
    data=[]
    if os.path.exists(cache_file):
        logging.info("Loading data from cache...")
        data = load(cache_file)
    else:
        print("No data to load.")
    
    # inputsALL, outputsALL_label = data
    inputsALL, outputsALL_label, output_name = data
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


    # outputsALL=outputsALL_label
    print(outputsALL)
    return inputsALL,outputsALL,output_name


def compute_class_weights(y_train):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights_dict = dict(enumerate(class_weights))
    # class_weights_dict = {0: 0.1, 1: 0.9}
    return class_weights_dict

def create_lstm_model(input_shape, output_shape, lstm_units1=160, lstm_units2=192, dropout_rate1=0.3, dropout_rate2=0.0, 
                      l2_regularizer1=0.0099, l2_regularizer2=0.0004):
    model = Sequential([
        LSTM(lstm_units1, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(l2_regularizer1)),
        Dropout(dropout_rate1),  # Dropout for regularization
        LSTM(lstm_units2, kernel_regularizer=l2(l2_regularizer2)),
        Dropout(dropout_rate2),  # Dropout for regularization
        Dense(output_shape, activation='softmax')
    ])
    return model

def compile_and_fit_model(model, X_train, y_train, X_test, y_test, learning_rate=0.008, epochs=150, validation_split=0.2, patience=3, class_weights=None):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
                        callbacks=[callback], class_weight=class_weights)
    return history
# def compute_class_weights(y_train):
#     class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
#     class_weights_dict = dict(enumerate(class_weights))
#     class_weights_dict = {0: 0.1, 1: 0.9}
#     return class_weights_dict

# def create_lstm_model(input_shape, output_shape, lstm_units1=160, lstm_units2=192, dropout_rate1=0.3, dropout_rate2=0.0, 
#                       l2_regularizer1=0.0099, l2_regularizer2=0.0004):
#     model = Sequential([
#         LSTM(lstm_units1, input_shape=input_shape, return_sequences=True, kernel_regularizer=l2(l2_regularizer1)),
#         Dropout(dropout_rate1),  # Dropout for regularization
#         LSTM(lstm_units2, kernel_regularizer=l2(l2_regularizer2)),
#         Dropout(dropout_rate2),  # Dropout for regularization
#         Dense(output_shape, activation='softmax')
#         # Dense(1, activation='sigmoid')
#     ])
#     return model

# def compile_and_fit_model(model, X_train, y_train, X_test, y_test, learning_rate=0.008, epochs=150, validation_split=0.2, patience=3, class_weights=None):
#     callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), 
#                   loss='sparse_categorical_crossentropy', 
#                   metrics=['accuracy'])
#     history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test),
#                         callbacks=[callback], class_weight=class_weights)
#     return history

def save_model(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model.save(os.path.join(output_dir, 'lstm_model_fft.keras'))

def save_training_params(params, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    params_path = os.path.join(output_dir, 'training_params.json')
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)

def create_incremented_folder(base_dir, base_name):
    count = 1
    new_folder = os.path.join(base_dir, f"{base_name}_{count}")
    while os.path.exists(new_folder):
        count += 1
        new_folder = os.path.join(base_dir, f"{base_name}_{count}")
    os.makedirs(new_folder)
    return new_folder


if __name__ == "__main__":
    args = parse_arguments()
    
    stability_label_dict = load_stability_labels(args.stability_file)
    filenames = get_filenames(args.data_root)

    cache_file=get_cache_file_path(args.project_root,args.window_size,args.approach)

    # # inputsALL, outputsALL = get_data(cache_file, filenames, stability_label_dict)
    inputsALL, outputsALL, output_name = get_data(cache_file, filenames, stability_label_dict)
    # print(f"shapes: {inputsALL.shape}, {outputsALL.shape} {output_name.shape}")

    # df_new_cluster = pd.read_csv("C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\labels\\h2kmeans_thr_0.7_label.csv")
    # new_outputSAll=np.zeros_like(outputsALL)
    # for i in range(len(output_name)):
    #     new_outputSAll[i]=df_new_cluster[df_new_cluster['filename']==output_name[i]]['cluster_label'].values[0]
    df_new_cluster = pd.read_csv("C:\\Users\\qpw475\\Documents\\combustion_instability\\data\\labels\\h2kmeans_thr_0.7_label.csv")
    new_outputSAll=np.zeros_like(outputsALL)
    
    for i in range(len(output_name)):
        # new_outputSAll[i]=df_new_cluster[df_new_cluster['filename']==output_name[i]]['cluster_label'].values[0]
        new_outputSAll[i]=df_new_cluster[df_new_cluster['filename']==output_name[i]]['label_stephany'].values[0]
    
    # new_outputSAll = new_outputSAll.astype(int)
    label_mapping = {'Stable': 0, 'Unstable': 1}
    # new_outputSAll = new_outputSAll.astype(int)
    # print(new_outputSAll)
    new_outputSAll = np.where(new_outputSAll == "Stable", 0, 
                 np.where(new_outputSAll == "Unstable", 1, new_outputSAll))

    # Convert the array to integers (if needed)
    new_outputSAll = new_outputSAll.astype(int)
    # new_outputSAll = new_outputSAll.astype(int)
    # # new_outputSAll[new_outputSAll == 1] = 4
    # # # new_outputSAll[new_outputSAll == 0] = 1
    # # new_outputSAll[new_outputSAll == 2] = 1 #0
    # # new_outputSAll[new_outputSAll == 4] = 2
    # 
    outputsALL=new_outputSAll
    print(new_outputSAll.shape)
    print(outputsALL.shape)

    # Ensure reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # print(inputsALL[0][0][1])
    # print(inputsALL[0][1][1])
    # print(inputsALL[0][2][1])
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(inputsALL, outputsALL, test_size=args.test_size, random_state=args.seed)
    
    selected_features = list(map(int, args.features.split(',')))
    
    X_train = X_train[:, :, selected_features]
    X_test = X_test[:, :, selected_features]

    input_shape = X_train.shape[1:]  # Shape of each sample (timesteps, features)
    output_shape = len(np.unique(outputsALL))  # Assuming binary classification (stable or unstable)
    print(input_shape, output_shape)
    # Compute class weights
    class_weights_dict = compute_class_weights(y_train)
    print(class_weights_dict)

    # Create and compile the model
    model = create_lstm_model(input_shape, output_shape, args.lstm_units1, args.lstm_units2, 
                              args.dropout_rate1, args.dropout_rate2, 
                              args.l2_regularizer1, args.l2_regularizer2)
    history = compile_and_fit_model(model, X_train, y_train, X_test, y_test, args.learning_rate, args.epochs, 
                                    args.validation_split, args.patience, class_weights=class_weights_dict)
    # Create a new folder for the current run
    output_dir = create_incremented_folder(os.path.join(args.project_root, "model", args.approach, f"window_{args.window_size}ms"), "model")
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist

    # Save the training parameters and metrics
    training_params = {
        'lstm_units1': args.lstm_units1,
        'lstm_units2': args.lstm_units2,
        'dropout_rate1': args.dropout_rate1,
        'dropout_rate2': args.dropout_rate2,
        'l2_regularizer1': args.l2_regularizer1,
        'l2_regularizer2': args.l2_regularizer2,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs,
        'validation_split': args.validation_split,
        'patience': args.patience,
        'class_weights': class_weights_dict,
        'selected_features': selected_features,
        'history': history.history
    }

    predictions = model.predict(inputsALL[:, :, selected_features])
    print(predictions)
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(outputsALL, predictions)
    print(accuracy)


    # X_train=inputsALL[0:100, :, selected_features] 
    # X_test=inputsALL[100:110, :, selected_features]
    # #Create the SHAP explainer
    # explainer = shap.GradientExplainer(model,X_train)

    # # Explain a few test instances
    # shap_values = explainer.shap_values(X_test)

    # # Convert to NumPy array for visualization if necessary
    # shap_values = np.array(shap_values)

    # # print(shap_values)
    # print("X_train shape:", X_train.shape)
    # print("X_test shape:", X_test.shape)
    # print("shap_values shape:", np.array(shap_values).shape)


    # shap_values_2d = np.sum(shap_values, axis=(1, 3))  # Sum over time steps and classes
    # print("shap_values_2d shape:", shap_values_2d.shape)  # Expected: (10, 2)

    # X_test_2d = np.mean(X_test, axis=1)  # Average over time steps
    # print("X_test_2d shape:", X_test_2d.shape)  # Expected: (10, 2)

    # # Summary plot
    # shap.summary_plot(shap_values_2d, X_test_2d)

    # index = 0  # Choose a sample to visualize

    # shap_values_single = np.sum(shap_values[index], axis=0)  # Sum over time steps
    # print("shap_values_single shape:", shap_values_single.shape)  # Expected: (2, 2)

    # X_test_single = np.mean(X_test[index], axis=0)  # Average over time
    # print("X_test_single shape:", X_test_single.shape)  # Expected: (2,)


    # shap_mean_importance = np.mean(np.abs(shap_values_2d), axis=0)
    # print(shap_mean_importance)

    # # shap.force_plot(explainer.expected_value, shap_values_2d[0], X_test_2d[0])

    # # Assuming shap_values has shape (samples, time_steps, features)
    # shap_values_abs = np.abs(shap_values_2d.values)  # Take absolute values

    # # Aggregate SHAP values over features (sum over axis 2)
    # shap_time_importance = shap_values_abs.mean(axis=(0, 2))  # Mean over samples and features

    # # Plot time step importance
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(len(shap_time_importance)), shap_time_importance, marker='o')
    # plt.xlabel("Time Step")
    # plt.ylabel("Mean Absolute SHAP Value")
    # plt.title("Importance of Time Steps in LSTM Model")
    # plt.grid()
    # plt.show()

    #     # Get indices of top 5 most important time steps
    # top_time_steps = np.argsort(shap_time_importance)[-5:][::-1]
    # print("Most important time steps:", top_time_steps)



    # shap.summary_plot(shap_values, X_test)



    # shap.force_plot(explainer.expected_value[0], shap_values[0], X_test)

    # shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
    #                                  base_values=explainer.expected_value[0], 
    #                                  data=X_test[0]))








    save_training_params(training_params, output_dir)
    # Save the model
    save_model(model, output_dir)
