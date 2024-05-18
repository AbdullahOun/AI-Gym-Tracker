# AI-Gym-Tracker

Pose Classification with TensorFlow and OpenCV
Project Overview

This project leverages deep learning to classify body poses from video data. The primary goal is to differentiate between "up" and "down" poses using OpenCV for video capture, MediaPipe for pose detection, and TensorFlow with Keras for training and deploying the classification model.
Features

    Data Collection: Capture video frames and extract pose landmarks using OpenCV and MediaPipe.
    Data Preprocessing: Normalize pose landmarks and encode class labels.
    Model Training: Build and train a neural network model using TensorFlow and Keras.
    Model Evaluation: Evaluate model performance using various metrics and visualize the training process.
    Real-Time Pose Classification: Utilize the trained model to classify poses in real-time and count the occurrences of "up" and "down" poses.

Project Structure

    data_collection.py: Script to capture video frames, extract pose landmarks, and save them to a CSV file.
    model_training.py: Script to preprocess data, train the neural network, and evaluate its performance.
    pose_classification.py: Script to classify poses in real-time using the trained model.
    coords.csv: CSV file containing the pose landmarks and their corresponding class labels.
    pose_classification_model03.h5: Trained model file.
    requirements.txt: List of dependencies required for the project.
    README.md: Project description and setup instructions.

Dependencies

    OpenCV
    MediaPipe
    TensorFlow
    Scikit-learn
    Pandas
    Matplotlib
    Numpy

Setup Instructions

    Clone the repository:
        git clone https://github.com/AbdullahOun/AI-Gym-Tracker.git
        cd pose-classification

    Install the required dependencies:
        pip install -r requirements.txt

        Data Collection:
        Run data_collection.py to capture video frames and save pose landmarks to coords.csv.
        Use 'u' and 'd' keys to label the poses as "up" and "down", respectively.

    Model Training:
        Run model_training.py to preprocess data, train the model, and evaluate its performance.
        Training history and evaluation metrics will be displayed and saved.

    Real-Time Pose Classification:
        Run pose_classification.py to classify poses in real-time using the webcam or a video file.

Data Collection Script (data_collection.py)

This script captures video frames, extracts pose landmarks using MediaPipe, and saves the coordinates to coords.csv under the specified class labels ('up' and 'down').
Usage

    Press 'u' to label the current pose as "up".
    Press 'd' to label the current pose as "down".
    Press 'q' to quit the video capture.

Model Training Script (model_training.py)

This script preprocesses the data, builds and trains a neural network, and evaluates its performance. The script uses early stopping to prevent overfitting and visualizes the training process.

Real-Time Pose Classification Script (pose_classification.py)

This script uses the trained model to classify poses in real-time from a video feed. It tracks the "up" and "down" poses and counts them, displaying the results on the video feed.

Conclusion

This project demonstrates a practical application of deep learning for real-time pose classification. The scripts provided enable you to collect data, train a model, and classify poses in real-time, making it a comprehensive example of using machine learning for video analysis tasks.

For further improvements, consider experimenting with different model architectures, adding more classes, or enhancing the data preprocessing steps.
