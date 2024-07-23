# Trigger Word Detection

## Overview

This project focuses on applying deep learning techniques to speech recognition. The primary goal was to gain hands-on experience in structuring a speech recognition project and to master the processes involved in synthesizing and processing audio recordings to create training and development datasets. By building and refining a model to detect the trigger word "activate," this project demonstrates practical applications of deep learning in the field of speech recognition.

___

## Table of Contents
1. [Data](#data)
1. [Spectrogram](#spectrogram)
1. [Generating a Single Training Example](#generating-a-single-training-example)
1. [Model Architecture](#model-architecture)
1. [Installation](#installation)
1. [Usage](#usage)

___

## Data

The dataset is organized into the following directories:

- **`data/external/positive`**: Contains audio recordings of people saying the word "activate."
- **`data/external/negative`**: Contains audio recordings of people saying random words other than "activate."
- **`data/external/background`**: Contains 10-second clips of background noise from various environments.

Each recording contains a single word or background noise.

___

## Spectrogram

A spectrogram is used to represent the audio data. 
It visualizes the frequencies present in the audio signal over time, calculated using a sliding window and Fourier transform.

### Spectrogram Details

- **Input to the Model**:
  - Number of time steps: 5511
  - Number of frequencies per time step: 101
- **Audio Clip Duration**: 10 seconds
- **Output of the Model**:
  - Number of units: 1375
  - The model predicts for each of the 1375 time steps whether the trigger word "activate" was recently said.

___

## Generating a Single Training Example

To create a single training example:

1. **Select a random 10-second background audio clip** from the `data/external/background` directory.
2. **Insert 0-4 clips of "activate"** from the `data/external/positive` directory into the background clip.
3. **Insert 0-2 clips of negative words** from the `data/external/negative` directory into the background clip.

### Labeling the Data

Labels $y^{\langle t \rangle}$ indicate whether the trigger word "activate" has recently been spoken:

- The label $y^{\langle t \rangle}$ is set to 0 for all time steps that do not contain the word "activate".
- For each inserted "activate" clip, the labels are set to 1 for 50 consecutive time steps after the end of the "activate" clip, indicating that the trigger word was recently spoken.

___


## Model Architecture

The goal is to build a network that ingests a spectrogram and outputs a signal when the trigger word "activate" is detected. The model architecture includes the following layers:

1. **1D Convolutional Layer**:
   - **Input**: 5511-step spectrogram, each step being a vector of 101 units.
   - **Output**: 1375-step output.
   - **Role**: Extracts low-level features and reduces the dimensionality from 5511 to 1375 time steps. This layer helps speed up the model by reducing the input size for the subsequent GRU layers.

2. **Two GRU Layers**:
   - **Purpose**: Process the sequence of inputs from left to right.

3. **Dense Layer with Sigmoid Activation**:
   - **Purpose**: Makes a prediction for $y^{\langle t \rangle}$. The sigmoid output estimates the probability of the output being 1, indicating the trigger word "activate" was recently spoken.
  
___

## Installation

1. Create a new environment with a 3.7 Python version.
1. Create a directory on your device and navigate to it.
1. Clone the repository:
   ```
   git clone https://github.com/amromeshref/Trigger-Word-Detection.git
   ```
1. Navigate to the Trigger-Word-Detection directory.
   ```
   cd Trigger-Word-Detection
   ```
1. Type the following command to install the requirements file using pip:
    ```
    pip install -r requirements.txt
    ```
___

## Usage

To detect the trigger word "activate" and superimpose a chime sound at the detected locations:
- Ensure the input audio file is in `wav` format.
```python
from src.predict_model import ModelPredictor

# Initialize the predictor
predictor = ModelPredictor()

# Example audio 
wav_file = "data/external/positive/3_act2.wav"

# Add a chime sound to the audio file at points where the word "activate" is detected
modified_audio = predictor.chime_on_activate(wav_file)

# Save the modified audio
modified_audio.export("modified_audio.wav", format="wav")
```



