# HarmoniGAN: Historical Audio Restoration via Melodic Optimization and Inpainting Networks

You can find the full code in our GitHub repository: https://github.com/johannasmriti/restoringAudio

This repository provides a deep learning model for denoising speech signals using a combination of **Autoencoder** and **Generative Adversarial Networks (GANs)**. The goal is to train a model that can take noisy speech as input and output clean speech.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [1. Download Dataset](#1-download-dataset)
- [Model Architecture](#model-architecture)
  - [1. Autoencoder Model](#1-autoencoder-model)
  - [2. GAN for Denoising](#2-gan-for-denoising)
- [Training the Model](#training-the-model)
  - [1. Preprocess the Audio Files](#1-preprocess-the-audio-files)
  - [2. Split the Data into Training and Validation Sets](#2-split-the-data-into-training-and-validation-sets)
  - [3. Fine-tune the Pre-trained Model](#3-fine-tune-the-pre-trained-model)
  - [4. Save the Fine-tuned Model](#4-save-the-fine-tuned-model)
- [Evaluating the Model](#evaluating-the-model)
- [Usage](#usage)
  - [1. Load the Trained Model](#1-load-the-trained-model)
  - [2. Denoise New Audio](#2-denoise-new-audio)
  - [3. Convert the Denoised Mel Spectrogram Back to Audio](#3-convert-the-denoised-mel-spectrogram-back-to-audio)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Project Overview

This project focuses on speech enhancement by training a model to remove noise from speech signals. It uses deep learning methods, including **Autoencoders** and **GANs**, to perform the denoising task.

---

## Requirements

Before running this project, ensure you have the following Python packages installed:

- `tensorflow`
- `keras`
- `librosa`
- `scikit-learn`
- `soundfile`

Install them using pip:

```bash
pip install tensorflow keras librosa scikit-learn soundfile
```

---

## Setup Instructions

### 1. Download Dataset

You will need a clean speech dataset and a noisy speech dataset to train the model. You can use any dataset that provides clean and noisy pairs, such as the VCTK corpus or other speech datasets.

```bash
# Example path where you store your datasets
# Clean audio path
PATH_TO_CLEAN = "../Dataset/train/train-clean/"
# Noisy audio path
PATH_TO_NOISY = "../Dataset/train/train-noisy/"
```

Ensure that the audio files are in `.wav` format.

---

### 2. Directory Structure

The project directory should look something like this:

```plaintext
Speech_DeNoiser_AE/
├── Dataset/
│   ├── train/
│   │   ├── train-clean/
│   │   ├── train-noisy/
├── trained_models/
│   ├── pretrained_model.h5  # Example pretrained model (can be autoencoder)
├── myModel/
│   ├── fine_tuned_model.h5  # Fine-tuned model after training
├── audio_denoiser.py  # Main script for audio denoising
└── README.md
```

---

## Model Architecture

### 1. Autoencoder Model

The core model is a **Convolutional Autoencoder**, where the encoder extracts important features from noisy spectrograms, and the decoder reconstructs the clean audio.

The autoencoder architecture is as follows:
- **Encoder**: A stack of convolutional layers with pooling to extract latent features.
- **Decoder**: A stack of convolutional layers with upsampling to reconstruct the audio.

### 2. GAN for Denoising

To improve denoising performance, we optionally use a **Generative Adversarial Network (GAN)**, where:

- **Generator**: A neural network that generates clean audio from noisy spectrograms.
- **Discriminator**: A neural network that distinguishes between real (clean) and generated (restored) spectrograms.

---

## Training the Model

### 1. Preprocess the Audio Files

Convert the audio files into **Mel spectrograms**. The `load_audio_files()` function takes care of loading the audio, extracting the Mel spectrograms, and resizing them.

```python
clean_data = load_audio_files(PATH_TO_CLEAN, target_shape=(1024, 44))
noisy_data = load_audio_files(PATH_TO_NOISY, target_shape=(1024, 44))
```

### 2. Split the Data into Training and Validation Sets

We split the dataset into training and validation sets using `train_test_split` from sklearn.

```python
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(noisy_data, clean_data, test_size=0.2, random_state=42)
```

### 3. Fine-tune the Pre-trained Model

If you have a pre-trained model (e.g., a previously trained autoencoder), fine-tune it with the noisy-clean pairs.

```python
model = load_pretrained_model(PATH_TO_PRETRAINED_MODEL)
fine_tune_model(model, x_train, y_train, x_val, y_val, epochs=20)
```

### 4. Save the Fine-tuned Model

Once training is complete, save the fine-tuned model.

```python
model.save(PATH_TO_FINE_TUNED_MODEL)
```

---

## Evaluating the Model

To evaluate the performance of the trained model, you can use metrics like **MSE** (Mean Squared Error) and **PSNR** (Peak Signal-to-Noise Ratio). These metrics help you understand how well the denoising process restores the clean speech signal.

```python
# Calculate PSNR
psnr_value = psnr(y_true, y_pred)
print(f"PSNR: {psnr_value}")

# Calculate NRMSE
nrmse_value = nrmse(y_true, y_pred)
print(f"NRMSE: {nrmse_value}")
```

---

## Usage

Once the model is trained and fine-tuned, you can use it to denoise new audio files.

### 1. Load the Trained Model

```python
model = load_pretrained_model(PATH_TO_FINE_TUNED_MODEL)
```

### 2. Denoise New Audio

To denoise new audio, you can use the model as follows:

```python
noisy_audio = load_audio("path/to/noisy_audio.wav")
mel_noisy = create_mel_spectrogram(noisy_audio)
denoised_audio = model.predict(mel_noisy)
```

### 3. Convert the Denoised Mel Spectrogram Back to Audio

Finally, convert the denoised Mel spectrogram back to audio and save it:

```python
denoised_audio_waveform = mel_to_audio(denoised_audio)
sf.write("denoised_audio.wav", denoised_audio_waveform, 16000)
```

---

## Acknowledgements

- **Librosa**: for audio processing and feature extraction.
- **TensorFlow**: for building and training the deep learning models.
- **Soundfile**: for saving and loading audio files.
