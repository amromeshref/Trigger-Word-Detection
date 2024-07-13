import os
import sys

# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),".."))
sys.path.append(REPO_DIR_PATH)


from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np


def get_wav_info(wav_file: str) -> tuple[int, np.ndarray]:
    """
    Get the sampling rate and data from a WAV file
    Args:
        wav_file (str): Path to the WAV file
    Returns:
        tuple: Sampling rate and data
        rate (int): Sampling rate. It represents the number of samples of audio carried per second and is measured in Hertz (Hz).
        data (numpy.ndarray): The audio data itself represented as a numpy array.
    """
    rate, data = wavfile.read(wav_file)
    return rate, data

def compute_spectrogram(wav_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the spectrogram of a WAV file
    Args:
        wav_file (str): Path to the WAV file
    Returns:
        tuple: Frequency, time and spectrogram
        freqs (numpy.ndarray): Array of sample frequency bins (in Hertz) corresponding to the rows of the spectrogram matrix sxx.
        times (numpy.ndarray): Array of time bins (in seconds) corresponding to the columns of the spectrogram matrix sxx.
        spectrogram_matrix (numpy.ndarray): Spectrogram matrix, where each entry represents the intensity (power) of a specific frequency at a specific time.
    """
    _, data = get_wav_info(wav_file)
    # Length of each window segment
    nfft = 200 
    # Sampling frequency
    fs = 8000 
    # Overlap between windows
    noverlap = 120 
    nchannels = data.ndim
    if nchannels == 1:
        freqs, times, spectrogram_matrix = spectrogram(data, fs=fs, nperseg=nfft, noverlap=noverlap)
    elif nchannels == 2:
        freqs, times, spectrogram_matrix = spectrogram(data[:,0], fs=fs, nperseg=nfft, noverlap=noverlap)
    return (freqs, times, spectrogram_matrix)

def plot_spectrogram(freqs: np.ndarray, times: np.ndarray, spectrogram_matrix: np.ndarray) -> None:
    """
    Plot the spectrogram
    Args:
        freqs (numpy.ndarray): Array of sample frequency bins (in Hertz) corresponding to the rows of the spectrogram matrix sxx.
        times (numpy.ndarray): Array of time bins (in seconds) corresponding to the columns of the spectrogram matrix sxx.
        spectrogram_matrix (numpy.ndarray): Spectrogram matrix, where each entry represents the intensity (power) of a specific frequency at a specific time.
    Returns:
        None
    """
    plt.pcolormesh(times, freqs, 10 * np.log10(spectrogram_matrix), shading='gouraud')
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [s]")
    plt.title("Spectrogram")
    plt.colorbar(label="Intensity [dB]")
    plt.show()