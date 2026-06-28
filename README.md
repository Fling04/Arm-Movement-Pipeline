# Arm-Movement-Pipeline

A data pipeline and machine learning system for classifying hand gestures using a sensor-instrumented glove, built as a foundation for prosthetic hand control.

## Overview
This capstone project focused on developing an end-to-end gesture recognition pipeline capable of classifying 21 distinct hand gestures from physical sensor data. The system captures motion data from a custom instrumented glove, processes it through a full cleaning and feature extraction pipeline, and trains multiple machine learning models to evaluate classification performance.

The pipeline integrates embedded firmware, signal processing, and machine learning to demonstrate that gesture classification is achievable from physical sensors alone, with EMG integration planned as a natural next step toward prosthetic control.

## Objectives
- Capture hand motion data from flex sensors and IMU at 400 Hz
- Clean and preprocess raw signals to remove noise and artifacts
- Extract meaningful features via sliding window analysis
- Train and compare classical and deep learning classifiers
- Demonstrate generalization across subjects
- Establish a pipeline ready for live inference deployment

## System Components
- Sensor-instrumented glove (5 flex sensors + IMU)
- Teensy microcontroller for data capture and streaming
- Signal preprocessing pipeline
- Feature extraction and windowing system
- Multi-model training and evaluation framework

## Technologies
- Python / C++
- Teensy 4.1
- TensorFlow / Keras
- scikit-learn
- JSON / NumPy

## Key Contributions
- Designed and built the full data capture and preprocessing pipeline
- Implemented sliding window feature extraction across 21 gesture classes
- Trained and compared KNN, SVM, and 1D CNN classifiers on feature vectors and raw sequences
- Evaluated within-subject and cross-subject generalization performance
- Established a pipeline architecture designed for live inference deployment

## Future Work
- EMG sensor integration for richer signal capture
- Expanded subject pool for improved generalization
- Lighter CNN architecture for real-time embedded inference
- Sideways finger flexion sensing
- Full live deployment and inference testing
