import os
import sys
# Add the repository directory path to the Python path
REPO_DIR_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(REPO_DIR_PATH)

from src.utils import load_config, compute_spectrogram
import matplotlib.pyplot as plt
import pydub
import numpy as np
import argparse


class DataTransformer():
    def __init__(self) -> None:
        self.config = load_config()
        self.background_max_time_ms = self.config["background_max_time_ms"]
        self.Ty = self.config["Ty"]

    def load_data(self) -> tuple[list[pydub.audio_segment.AudioSegment], list[pydub.audio_segment.AudioSegment], list[pydub.audio_segment.AudioSegment]]:
        """
        Load the audio data
        Args:
            None
        Returns:
            tuple: Background, positive and negative audio clips
            backgrounds (list): List of background audio clips in the form of pydub.AudioSegment objects
            positives (list): List of positive audio clips in the form of pydub.AudioSegment objects
            negatives (list): List of negative audio clips in the form of pydub.AudioSegment objects
        """
        positives = []
        negatives = []
        backgrounds = []

        DATA_DIR_PATH = os.path.join(REPO_DIR_PATH, "data/external")

        for filename in os.listdir(DATA_DIR_PATH + "/positive"):
            if filename.endswith(".wav"):
                positives.append(pydub.AudioSegment.from_wav(
                    DATA_DIR_PATH + "/positive/" + filename))

        for filename in os.listdir(DATA_DIR_PATH + "/negative"):
            if filename.endswith(".wav"):
                negatives.append(pydub.AudioSegment.from_wav(
                    DATA_DIR_PATH + "/negative/" + filename))

        for filename in os.listdir(DATA_DIR_PATH + "/background"):
            if filename.endswith(".wav"):
                backgrounds.append(pydub.AudioSegment.from_wav(
                    DATA_DIR_PATH + "/background/" + filename))

        return backgrounds, positives, negatives

    def get_random_time_segment(self, segment_duration_ms: int) -> tuple[int, int]:
        """
        Gets a random time segment of duration segment_duration_ms in a 10,000 ms audio clip.
        Args:
            segment_duration_ms (int): Duration of the audio clip in milliseconds
        Returns:
            tuple: Start and end time of the audio segment
            segment_start (int): Start time of the audio segment in ms
            segment_end (int): End time of the audio segment in ms
        """
        segment_start = np.random.randint(
            low=0, high=self.background_max_time_ms-segment_duration_ms)
        segment_end = segment_start + segment_duration_ms - 1
        return (segment_start, segment_end)

    def is_overlapping(self, segment1: tuple[int, int], segment2: tuple[int, int]) -> bool:
        """
        Check if two segments overlap
        Args:
            segment1 (tuple): Start and end time of the first segment in ms
            segment2 (tuple): Start and end time of the second segment in ms
        Returns:
            bool: True if the segments overlap, False otherwise
        """
        s1_start, s1_end = segment1
        s2_start, s2_end = segment2
        return s1_start <= s2_end and s2_start <= s1_end

    def insert_audio_clip(self, background: pydub.audio_segment.AudioSegment, audio_clip: pydub.audio_segment.AudioSegment, segment_time_start_ms: int) -> pydub.audio_segment.AudioSegment:
        """
        Insert an audio clip on a background audio.
        Args:
            background (pydub.audio_segment.AudioSegment): Background audio
            audio_clip (pydub.audio_segment.AudioSegment): Audio clip to be inserted
            segment_time_start_ms (int): Start time of the audio segment in ms
        Returns:
            pydub.audio_segment.AudioSegment: Audio clip inserted on the background audio
        """
        background = background.overlay(
            audio_clip, position=segment_time_start_ms)
        return background

    def label_training_example(self, y: np.ndarray, segment_end_ms: int) -> np.ndarray:
        """
        Insert additional ones in the label vector y
        Args:
            y (numpy.ndarray): Label vector of shape (1, Ty)
            segment_end_ms (int): End time of the audio segment in ms
        Returns:
            numpy.ndarray: Label vector with additional ones of shape (1, Ty)
        """
        # duration of the background (in terms of spectrogram time-steps)
        segment_end_y = int(segment_end_ms * self.Ty /
                            self.background_max_time_ms)
        # Add 1 to the number of time steps corresponding to the duration in the background
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < y.shape[1]:
                y[0, i] = 1
        return y

    def create_training_example(self, backgrounds: pydub.audio_segment.AudioSegment, positives: list[pydub.audio_segment.AudioSegment], negatives: list[pydub.audio_segment.AudioSegment]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pydub.audio_segment.AudioSegment]:
        """
        Create a training example
        Args:
            backgrounds (list): List of background audio clips
            positives (list): List of positive audio clips
            negatives (list): List of negative audio clips
        Returns:
            tuple: Frequency, time, spectrogram, label vector and background audio
            freqs (numpy.ndarray): Array of sample frequency bins (in Hertz) corresponding to the rows of the spectrogram matrix sxx.
            times (numpy.ndarray): Array of time bins (in seconds) corresponding to the columns of the spectrogram matrix sxx.
            X (numpy.ndarray): Spectrogram matrix, where each entry represents the intensity (power) of a specific frequency at a specific time.
            y (numpy.ndarray): Label vector of shape (1, Ty)
            background (pydub.audio_segment.AudioSegment): 10 second Audio clip with inserted positive and negative audio clips
        """
        # Select a random background audio clip
        random_index = np.random.randint(len(backgrounds))
        background = backgrounds[random_index]

        # Make background quieter
        background = background - 20

        previous_segments = []

        # Initialize y (label vector) of zeros
        y = np.zeros((1, self.Ty))

        # Select 0-4 random "positive" audio clips from the list of "positives"
        number_of_positives = np.random.randint(0, 5)

        # Select 0-2 random "negative" audio clips from the list of "negatives"
        number_of_negatives = np.random.randint(0, 3)

        for _ in range(number_of_positives):
            # Select a random positive audio clip
            random_index = np.random.randint(len(positives))
            random_positive = positives[random_index]

            # Get the duration of the audio clip in milliseconds
            audio_duration_ms = int(random_positive.duration_seconds * 1000)

            # Check if the audio clip is overlapping with previous segments
            if len(previous_segments) > 0:
                while True:
                    segment_time = self.get_random_time_segment(
                        audio_duration_ms)
                    OVERLAPPING = False
                    for segment in previous_segments:
                        if self.is_overlapping(segment_time, segment):
                            OVERLAPPING = True
                            break
                    if not OVERLAPPING:
                        break

            else:
                segment_time = self.get_random_time_segment(audio_duration_ms)

            previous_segments.append(segment_time)

            # Insert the audio clip on the background
            background = self.insert_audio_clip(
                background, random_positive, segment_time[0])

            # Label the training example by setting the labels to 1
            _, segment_end = segment_time
            y = self.label_training_example(y, segment_end)

        for _ in range(number_of_negatives):
            # Select a random negative audio clip
            random_index = np.random.randint(len(negatives))
            random_negative = negatives[random_index]

            # Get the duration of the audio clip in milliseconds
            audio_duration_ms = int(random_negative.duration_seconds * 1000)

            # Check if the audio clip is overlapping with previous segments
            if len(previous_segments) > 0:
                while True:
                    segment_time = self.get_random_time_segment(
                        audio_duration_ms)
                    OVERLAPPING = False
                    for segment in previous_segments:
                        if self.is_overlapping(segment_time, segment):
                            OVERLAPPING = True
                            break
                    if not OVERLAPPING:
                        break
            else:
                segment_time = self.get_random_time_segment(audio_duration_ms)

            previous_segments.append(segment_time)

            # Insert the audio clip on the background
            background = self.insert_audio_clip(
                background, random_negative, segment_time[0])

        # Export the modified background audio to a temporary WAV file
        WAV_FILE_PATH = os.path.join(REPO_DIR_PATH, "temp.wav")
        background.export(WAV_FILE_PATH, format="wav")

        # Compute the spectrogram of the modified background audio
        freqs, times, X = compute_spectrogram(WAV_FILE_PATH)

        # Remove the temporary WAV file
        os.remove(WAV_FILE_PATH)

        return (freqs, times, background, X, y)

    def generate_training_data(self, num_examples: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Generate training data
        Args:
            num_examples (int): Number of training examples to generate
        Returns:
            list: List of training examples in the form of tuples (X, y)
        """
        backgrounds, positives, negatives = self.load_data()
        training_data = []
        for _ in range(num_examples):
            _,_,_, X, y = self.create_training_example(
                backgrounds, positives, negatives)
            training_data.append((X, y))
        return training_data

    def save_training_data(self, training_data: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """
        Save the training data in the form of numpy arrays
        Args:
            training_data (list): List of training examples in the form of tuples (X, y)
        Returns:
            None
        """
        training_data = np.array(training_data, dtype=object)
        np.save(os.path.join(REPO_DIR_PATH,
                "data/processed/training_data.npy"), training_data)

    def visualize_labels(self, y: np.ndarray) -> None:
        """
        Visualize the labels
        Args:
            y (numpy.ndarray): Label vector of shape (1, Ty)
        Returns:
            None
        """
        # Make a copy of the label vector
        y_copy = y.copy()

        # Flatten the y array
        y_copy = y_copy.flatten()

        # Generate the time axis (x-axis)
        time = range(len(y_copy))

        # Plot the data
        plt.step(time, y_copy, where='post', linestyle='-')

        # Add labels and title
        plt.xlabel('Time')
        plt.ylabel('Label')

        # Add a grid for better readability
        plt.grid(True)

        # Display the plot
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=1000,
                        help="Number of training examples to generate")
    args = parser.parse_args()

    data_transformer = DataTransformer()
    training_data = data_transformer.generate_training_data(
        num_examples=args.num_examples)
    data_transformer.save_training_data(training_data)
    print("Training data saved successfully at data/processed/training_data.npy")