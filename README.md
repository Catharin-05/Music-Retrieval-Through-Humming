# Music Retrieval Through Humming

## Overview
This project aims to build a music retrieval system that matches hummed audio clips to Indian songs. Using machine learning and audio processing techniques such as spectrogram extraction and feature matching, the system allows users to search for songs based on their humming.

## Key Features
- **Spectrogram Extraction**: Converts audio clips into time-frequency representations for feature extraction.
- **Fingerprinting Algorithms**: Uses advanced matching techniques such as Dynamic Time Warping (DTW) and cross-correlation to match hummed audio to song files.
- **Machine Learning Model**: Implements supervised learning for song classification based on hummed features.
- **Scalable Architecture**: Designed to handle large datasets and can be expanded for global music retrieval with the addition of deep learning techniques.

## Project Structure

music_retrieval_through_humming/
|
├── src/
│   ├── feature_extraction.py   # Code for feature extraction from audio.
│   ├── model_training.py       # Code for training the retrieval model.
│   └── matching_algorithm.py   # Code for song matching based on features.
│
└── README.md                   # Project documentation.


## Dependencies
- `librosa` for audio processing.
- `sklearn` for machine learning model training.
- `numpy` for numerical operations.
- `matplotlib` for plotting and visualization.

You can install these dependencies using `pip`:

```bash
pip install librosa scikit-learn numpy matplotlib
