# Gesture Recognition

## Overview

This project focuses on building a **gesture recognition system** using machine learning, specifically LSTM models, to predict and identify gestures in real-time. The project comprises three main components: **data collection**, **model building**, and **real-time detection**.

## Dependencies

### Python Libraries
- **mediapipe:** Utilized for holistic human pose estimation, including facial, pose, and hand landmarks.
- **cv2:** OpenCV for computer vision tasks, such as video capture and image processing.
- **matplotlib + seaborn:** Used for data visualization and visual evaluation.
- **sklearn:** Used for model evaulation after training.
- **numpy:** Utilized for various numerical tasks and array manipulation.
- **tensorflow:** Deep learning framework for building and training machine learning models.

### Project-Specific Modules
- **utils.config:** Configuration file containing constants and parameters used across the project.
- **utils.mp_helper:** Module containing helper functions for working with the Mediapipe library.

## Important Files

### 1. **gesture_collect_data.ipynb**
This notebook is responsible for collecting training data through webcam capture and Mediapipe landmark detection. Key components include detection functions, data extraction functions, and the main data collection process.

### 2. **gesture_model_build.ipynb**
This notebook focuses on building an LSTM model for gesture recognition. It covers constants definition, loading data from the collected dataset, model architecture definition, training, and model evaluation.

### 3. **real_time_detection.ipynb**
In this notebook, a pre-trained gesture recognition model is loaded, and real-time detection is performed using the webcam. It includes functions for probability visualization and a loop for continuous real-time gesture detection.

## Workflow

1. **Data Collection:**
   - Use `gesture_collect_data.ipynb` to capture video frames, perform Mediapipe pose estimation, and save keypoints for each frame.
   - Define actions, sequences, and frames for data collection.
   - Add more actions by modifiyng the config file.

2. **Model Building:**
   - Utilize `gesture_model_build.ipynb` to load the collected data.
   - Build an LSTM model for gesture recognition.
   - Train the model using the training dataset.
   - Evaluate the model on the test set.

3. **Real-Time Detection:**
   - Employ `real_time_detection.ipynb` to load the pre-trained model.
   - Continuously capture frames from the webcam.
   - Make real-time predictions and display results.
   - Stabilize predictions for smoother user experience.

## Notes
- Adjust paths and dependencies based on your specific project structure.
- The real-time detection loop can be further integrated into an application or used as a standalone tool for gesture recognition.
