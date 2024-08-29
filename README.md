# Deepfake Detection with InceptionResNetV2 and BiLSTM
The "InceptionResNet-BiLSTM Deepfake Detection" project provides a sophisticated deep learning framework that combines InceptionResNetV2 and Bidirectional Long Short-Term Memory (BiLSTM) networks to enhance the accuracy of deepfake video detection. This model addresses the increasing prevalence of deepfake content in digital media by offering a robust and scalable solution for detecting manipulated videos, contributing to the ongoing efforts to safeguard the authenticity and integrity of multimedia content.

## Repository Structure

- **[Building Model and Testing](https://github.com/fromjyce/DeepfakeDetection/tree/main/BuildingModelandTesting)**
  - **PreProcessing.ipynb**: Code for video preprocessing using OpenCV and MTCNN for face detection and resizing, with parallel processing using `ThreadPoolExecutor`.
  - **InceptionResNet-BiLSTMModelBuild.ipynb**: Custom model combining InceptionResNetV2 (for feature extraction) and BiLSTM (for temporal patterns), with layers for dense, dropout, LSTM, and softmax-based prediction.
  - **TesterCode.ipynb**: Code to test the saved model to classify input videos as real or fake.

## Methodology

- **Dataset**: Utilized [Kaggle’s Deepfake Detection Challenge dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) with 400 videos and corresponding labels.
- **Preprocessing**: Frames extracted using OpenCV, faces detected with MTCNN, resized to 300x300, and processed in parallel using `ThreadPoolExecutor`.
- **Model Architecture**: Hybrid InceptionResNetV2 (pre-trained on ImageNet) for visual feature extraction, integrated with BiLSTM layers to capture temporal dependencies. The model employs dense layers and softmax for final classification.
- **Majority Voting Rule**: Frames are classified as real or fake, and the video is categorized based on the majority of frame classifications.

## Running of the Programs
The programs have been tested on Google Colab for training and testing, while I used the Visual Studio Code IDE in Windows 11 to develop a website for this model. ***You are free to choose any IDE that suits your needs.***

## Contact
If you come across any mistakes in the programs or have any suggestions for improvement, please feel free to contact me <jaya2004kra@gmail.com>. I appreciate any feedback that can help me improve my coding skills

## License
All the programs in this repository are licensed under the GPL-3.0 license. You can use them for educational purposes and modify them as per your requirements. ***However, I do not take any responsibility for the accuracy or reliability of the programs.***

## MY SOCIAL PROFILES:
### [LINKEDIN](https://www.linkedin.com/in/jayashrek/)

