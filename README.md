# Speech Denoising using Autoencoders and GANs

This project involves training a neural network model for denoising audio signals, specifically aimed at removing noise from speech recordings. The model is based on a combination of Convolutional Autoencoders (AE) and Generative Adversarial Networks (GANs) for improved audio restoration.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
  - [1. Download Dataset](#1-download-dataset)
  - [2. Directory Structure](#2-directory-structure)
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

This project aims to develop a speech denoising system using deep learning techniques. The model employs an autoencoder for noise removal and optionally fine-tunes a pre-trained model to improve performance. The denoising model operates on Mel spectrograms of audio, leveraging both supervised and unsupervised learning for noise reduction.

Key steps:
1. **Preprocessing**: Audio files are loaded, transformed into Mel spectrograms, and then resized and padded to a fixed shape.
2. **Model Training**: A pre-trained autoencoder is fine-tuned with clean and noisy audio pairs.
3. **Evaluation**: The model's performance is assessed using PSNR (Peak Signal-to-Noise Ratio) and MSE (Mean Squared Error).
4. **Restoration**: The denoised audio is reconstructed from the predicted Mel spectrogram.

---

## Requirements

The following libraries are required to run the code:

- `librosa` (for audio processing and feature extraction)
- `tensorflow` (for deep learning)
- `numpy` (for numerical operations)
- `scikit-learn` (for model evaluation and dataset splitting)
- `matplotlib` (for visualization)
- `soundfile` (for saving audio files)

Install the required dependencies using:

```bash
pip install librosa tensorflow numpy scikit-learn matplotlib soundfile
