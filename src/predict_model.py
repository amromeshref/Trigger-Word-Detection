import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(REPO_DIR_PATH)

from src.utils import compute_spectrogram
from tensorflow.keras.models import model_from_json
import tensorflow as tf
import numpy as np
import pydub
from typing import Tuple

THRESHOLD = 0.5


class ModelPredictor():
    def __init__(self):
        self.model = self.load_model()
    
    def load_model(self) -> tf.keras.Model:
        """
        Load the pre-trained model
        Args:
            None
        Returns:
            tf.keras.Model: Pre-trained model
        """
        json_file_path = os.path.join(REPO_DIR_PATH, "models", "best-models", "model.json")
        json_file = open(json_file_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        model_path = os.path.join(REPO_DIR_PATH, "models", "best-models", "model.h5")
        model.load_weights(model_path)
        return model
    
    def ensure_input_10_seconds(self, audio: pydub.audio_segment.AudioSegment) -> pydub.audio_segment.AudioSegment:
        """
        Ensure that the audio is 10 seconds long
        Args:
            audio (pydub.audio_segment.AudioSegment): Audio segment
        Returns:
            pydub.audio_segment.AudioSegment: Audio segment that is 10 seconds long
        """
        input_duration = audio.duration_seconds
        if input_duration == 10.0:
            return audio
        elif input_duration < 10.0:
            silence = pydub.AudioSegment.silent(duration=(10.0 - input_duration)*1000+1000)
            result = audio + silence
            result = result[:10000]
            return result
        else:
            return audio[:10000]
    
    def data_preprocessing(self, wav_file: str) -> Tuple[pydub.audio_segment.AudioSegment, np.ndarray]:
        """
        Preprocess the data
        Args:
            wav_file (str): Path to the WAV file
        Returns:
            tuple: Audio and input data
            audio (pydub.audio_segment.AudioSegment): Modified Audio segment that is 10 seconds long
            X (numpy.ndarray): Input data of shape (1, Ty, n_freq)
        """
        # Load the WAV file
        audio = pydub.AudioSegment.from_wav(wav_file)

        # Ensure that the audio is 10 seconds long
        audio = self.ensure_input_10_seconds(audio)

        # Export the audio to a temporary WAV file
        WAV_FILE_PATH = os.path.join(REPO_DIR_PATH, "temp.wav")
        audio.export(WAV_FILE_PATH, format="wav")

        # Compute the spectrogram (the input to the model)
        _, _, X = compute_spectrogram(WAV_FILE_PATH)
        X  = X.swapaxes(0,1)
        X = np.expand_dims(X, axis=0)

        # Remove the temporary WAV file
        os.remove(WAV_FILE_PATH)

        return (audio,X)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the output of the model
        Args:
            X (numpy.ndarray): Input data of shape (num_samples, Ty, n_freq)
        Returns:
            numpy.ndarray: Output data of shape (num_samples, Ty, 1)
        """
        return self.model.predict(X)
    
    def predict(self, wav_file: str) -> Tuple[pydub.audio_segment.AudioSegment, np.ndarray]:
        """
        Predict the output of the model
        Args:
            wav_file (str): Path to the WAV file
        Returns:
            tuple: Audio and output data
            audio (pydub.audio_segment.AudioSegment): Modified Audio segment that is 10 seconds long
            predictions (numpy.ndarray): Output data of shape (1, Ty, 1)
        """
        audio, X = self.data_preprocessing(wav_file)
        return (audio, self.model.predict(X))
    
    def chime_on_activate(self, wav_file: str):
        audio_clip = pydub.AudioSegment.from_wav(wav_file)
        chime_file_path = os.path.join(REPO_DIR_PATH, "data", "external", "chime.wav")
        chime = pydub.AudioSegment.from_wav(chime_file_path)
        audio_clip, predictions = self.predict(wav_file)

        Ty = predictions.shape[1]
        # Step 1: Initialize the number of consecutive output steps to 0
        consecutive_timesteps = 0
        i = 0
        # Step 2: Loop over the output steps in the y
        while i < Ty:
            # Step 3: Increment consecutive output steps
            consecutive_timesteps += 1
            # Step 4: If prediction is higher than the threshold for 20 consecutive output steps have passed
            if consecutive_timesteps > 20:
                # Step 5: Superpose audio and background using pydub
                audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
                # Step 6: Reset consecutive output steps to 0
                consecutive_timesteps = 0
                i = 75 * (i // 75 + 1)
                continue
            # if amplitude is smaller than the threshold reset the consecutive_timesteps counter
            if predictions[0, i, 0] < THRESHOLD:
                consecutive_timesteps = 0
            i += 1
        return audio_clip