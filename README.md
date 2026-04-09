# NNDL-Lab-code  
Neural Networks and Deep Learning (NNDL) Lab  

## Overview
This repository contains implementations of core Deep Learning experiments performed as part of the NNDL lab. Each experiment applies neural network models using TensorFlow and Keras to solve real-world problems across computer vision and time-series domains.

## Objectives
- Understand deep learning concepts through practical implementation  
- Work with architectures like ANN, CNN, RNN, and LSTM  
- Apply models on real-world datasets  
- Evaluate and interpret model performance using metrics and visualizations  

## Experiments

### 1. Face Recognition Attendance System
This experiment uses OpenCV for face detection and feature extraction. Extracted features are classified using an SVM model. The system captures real-time video input, identifies individuals, and logs attendance automatically. It demonstrates integration of computer vision with machine learning for automation.

### 2. Fraud Detection using ANN
A fully connected neural network is trained on transaction data to classify fraud. Data preprocessing includes normalization and handling class imbalance using techniques such as class weights. The model uses ReLU and Sigmoid activation functions and is evaluated using accuracy and confusion matrix.

### 3. MNIST Digit Classification
A neural network (ANN/CNN) is trained on grayscale digit images. The pipeline includes normalization, reshaping, and one-hot encoding. The model learns digit patterns and is evaluated using accuracy, with training and validation loss plotted to analyze convergence.

### 4. CNN Image Classification
A CNN model with convolution, pooling, and dense layers is used for image classification. Feature extraction is performed automatically through filters. The model captures spatial hierarchies and improves classification accuracy.

### 5. RNN Temperature Prediction
A sequential model using Simple RNN layers is trained on time-series temperature data. Input sequences are created using sliding windows. The model learns temporal dependencies and predicts future values.

### 6. LSTM Autoencoder for Anomaly Detection
An LSTM autoencoder is trained to reconstruct input sequences. Reconstruction error is used to identify anomalies. A threshold is defined to separate normal and abnormal patterns.

### 7. Object Detection (Smart Parking System)
This experiment detects vehicles using deep learning-based object detection techniques. Bounding boxes are drawn around detected objects. It demonstrates applications in smart parking and traffic monitoring.

### 8. CIFAR-10 Classification
A deep CNN is trained on the CIFAR-10 dataset. The model includes multiple convolutional and pooling layers followed by dense layers. Feature map visualization helps understand learned representations.

### 9. Pneumonia Detection (Transfer Learning)
MobileNetV2 is used as a pre-trained model with fine-tuning. Only top layers are retrained. This improves accuracy and reduces training time, especially for medical image classification.

### 10. Real-Time Object Detection (YOLO)
YOLO is used for detecting objects in real-time video streams. The model processes frames in a single pass and outputs bounding boxes with confidence scores and FPS.

### 11. (Not Included)
The GAN (Generative Adversarial Network) experiment is not included in this repository.

### 12. Time Series Forecasting (LSTM)
An LSTM model is trained on sequential data to forecast future values. Data is scaled and reshaped appropriately. Model performance is evaluated using MAE and RMSE.

## Technologies Used
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy, Pandas  
- Matplotlib  
- KaggleHub  

## Evaluation Metrics
- Accuracy  
- Confusion Matrix  
- Mean Squared Error (MSE)  
- Root Mean Squared Error (RMSE)  
- Mean Absolute Error (MAE)  
- R² Score  

## How to Run
pip install tensorflow keras numpy pandas matplotlib opencv-python kagglehub

Run the notebooks in Jupyter or Google Colab using Shift + Enter. Ensure internet connection for dataset downloads.

## Notes
- GPU is recommended for faster training  
- Some experiments may take longer on CPU  
- Ensure correct dataset paths  
- GAN experiment is not included  

## Conclusion
This lab provides hands-on experience with deep learning techniques and demonstrates their application in real-world scenarios such as image classification, anomaly detection, and time-series forecasting.
