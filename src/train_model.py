import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(REPO_DIR_PATH)

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, GRU, Dropout, BatchNormalization, TimeDistributed, Activation
from tensorflow.keras.optimizers import Adam


class ModelTrainer:
    def __init__(self):
        pass

    def load_data(self):
        """
        Load the training data from the processed data directory
        Args:
            None
        Returns:
            numpy.ndarray: Training data
        """
        DATA_PATH = os.path.join(REPO_DIR_PATH, "data", "processed", "training_data.npy")
        data = np.load(DATA_PATH, allow_pickle=True)
        return data
    
    def split_data(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets
        Args:
            data (numpy.ndarray): Training data
        Returns:
            tuple: Training and testing data
            X_train (numpy.ndarray): Training input data
            X_test (numpy.ndarray): Testing input data
            y_train (numpy.ndarray): Training output data
            y_test (numpy.ndarray): Testing output data
        """
        # Split the data into input and output
        X = [d[0] for d in data]
        Y = [d[1] for d in data]

        # Reshape the data and split it into training and testing sets
        X = np.array(X)
        X = X.reshape((X.shape[0],X.shape[2],X.shape[1]))
        Y = np.array(Y)
        Y = Y.reshape((len(data),Y.shape[2],1))
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        return (X_train, X_test, y_train, y_test)
    
    def create_model(input_shape: tuple[int,int]) -> tf.keras.Model:
        """
        Create the model
        Args:
            input_shape (tuple): Shape of the input data
        Returns:
            tf.keras.Model: The model
        """
        model = Sequential(
            [
                Conv1D(filters=196, kernel_size=15, strides=4, input_shape = input_shape),
                BatchNormalization(),
                Activation('relu'),
                Dropout(rate=0.8),
                GRU(128, return_sequences=True),
                Dropout(rate=0.8),
                BatchNormalization(),                          
                GRU(128, return_sequences=True),
                Dropout(rate=0.8),    
                BatchNormalization(),
                Dropout(rate=0.8),                              
                # Applies a dense (fully connected) layer to each time step independently.  
                TimeDistributed(Dense(1, activation='sigmoid')) 
            ]
        )
        return model
    
    