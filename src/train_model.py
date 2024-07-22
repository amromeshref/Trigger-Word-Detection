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
from datetime import datetime
import argparse

# Define the default hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 64

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
    
    def create_model(self, input_shape: tuple[int,int]) -> tf.keras.Model:
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
    
    def train(self, batch_size: int = BATCH_SIZE, epochs: int = NUM_EPOCHS) -> None:
        """
        Train the model
        Args:
            batch_size (int): Batch size
            epochs (int): Number of epochs
        Returns:
            None
        """
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.split_data(data)
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = self.create_model(input_shape)
        opt = Adam(learning_rate=1e-6, beta_1=0.9, beta_2=0.999)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])    
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        current_time = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")
        model.save(os.path.join(REPO_DIR_PATH, "models", "trained-models", f"model_{current_time}.h5"))
        print("Model trained and saved successfully at", os.path.join(REPO_DIR_PATH, "models", "trained-models", f"model_{current_time}.h5"))
        print("Model evaluation:")
        print(model.evaluate(X_test, y_test))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    args = parser.parse_args()
    trainer = ModelTrainer()
    trainer.train(args.batch_size, args.epochs)
