# Trigger Word Detection

## Overview

This project involves developing an AI model to detect the trigger word "activate" from audio recordings. 
The model is designed to analyze spectrogram representations of the audio data to determine if the trigger word is present.

## Data

The dataset is organized into the following directories:

- **`data/external/positive`**: Contains audio recordings of people saying the word "activate."
- **`data/external/negative`**: Contains audio recordings of people saying random words other than "activate."
- **`data/external/background`**: Contains 10-second clips of background noise from various environments.

Each recording contains a single word or background noise.

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

## Generating a Single Training Example

To create a single training example:

1. **Select a random 10-second background audio clip** from the `data/external/background` directory.
2. **Insert 0-4 clips of "activate"** from the `data/external/positive` directory into the background clip.
3. **Insert 0-2 clips of negative words** from the `data/external/negative` directory into the background clip.

### Labeling the Data

Labels $y^{\langle t \rangle}$ indicate whether the trigger word "activate" has recently been spoken:

- The label $y^{\langle t \rangle}$ is set to 0 for all time steps, as it does not contain "activate".
- For each inserted "activate" clip, update the labels for 50 consecutive time steps after the end of the "activate" clip to 1.
- This means the label for time steps immediately following the end of the "activate" clip is set to 1, indicating that the trigger word was recently spoken.

#### Example

Suppose you have a 10-second background clip, and an "activate" clip ends at the 5-second mark:
- **Convert 5 seconds into time steps**:
  - Given that 10 seconds correspond to 1375 time steps, 5 seconds correspond to timestep 687 (`int(1375 * (5 / 10))`).
- **Set Labels**:
  - Set $y^{\langle 688 \rangle}$ to $y^{\langle 737 \rangle}$ to 1, covering the 50 consecutive time steps after the end of the "activate" clip.

This method ensures that the training examples are varied and represent real-world scenarios where the trigger word might be spoken among other background noises or random words. It also helps to balance the training data by providing consistent labels for detection.



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
  


## Usage

1. **Prepare the Data**: Organize the audio files into their respective directories.
2. **Generate Spectrograms**: Convert audio files into spectrogram representations.
3. **Synthesize Training Examples**: Create and label training examples using the provided process.
4. **Train the Model**: Use the synthesized and labeled data to train the sequence model.
5. **Evaluate and Test**: Assess the model's performance on validation and test datasets.


## Requirements

- Python 3.x
- Required libraries: (List any additional libraries or dependencies)


