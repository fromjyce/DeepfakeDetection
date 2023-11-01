# Deepfake Detection with InceptionResNetV2 and BiLSTM model
Welcome to my GitHub repository for the "InceptionResNet-BiLSTM Deepfake Detection" project. In this repository, I've developed a cutting-edge deep learning framework that combines InceptionResNetV2 and Bidirectional Long-Short Term Memory (BiLSTM) networks to significantly improve the accuracy of deepfake video detection. My goal with this project is to address the growing concerns related to deepfake content in multimedia and provide a robust solution for its detection. By merging these two powerful technologies, I aim to contribute to the ongoing efforts in ensuring the integrity of digital media and protecting against deceptive content. I welcome collaboration, contributions, and exploration of this project to advance the field of deepfake detection.

## Structure of The Repository

## Methodology
### Dataset Utilized
I employed a [Kaggle dataset](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) that was originally provided for the [Deepfake Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge/overview) competition to train my model. This dataset comprised 400 videos along with a JSON file containing detailed information for each video, including labels indicating whether the video is genuine or fake. In cases where the video was identified as fake, the original video's name was also documented as part of the dataset.

### Preprocessing

In the preprocessing step, I extracted frames from the videos using the OpenCV library and performed face detection. To detect faces, I utilized the Multi-Task Cascaded Convolutional Neural Network (MTCNN). My main focus was on the faces within the video frames, as these regions are where actual manipulations take place. I then resized the extracted faces to 300 × 300 with three color channels. These resized facial images were used as input to my customized InceptionResNet-BiLSTM model, allowing me to compute effective deep features and classify the video as either real or fake. This meticulous preprocessing is a crucial step in enhancing the accuracy of my deepfake detection system.

In order to expedite the time-intensive preprocessing step, I harnessed the power of parallelization by employing Python's ThreadPoolExecutor with a specified number of worker threads, as determined by the `NUM_WORKERS` parameter. This parallel processing approach allowed multiple video files to undergo frame extraction, face detection, and resizing concurrently, significantly reducing the time required for preprocessing. The frames were then accumulated into the `data` list, and label sequences were generated and added to the `labels` list, enhancing the efficiency and overall performance of the deepfake detection system.

### Architecture Details

In this project, I've designed a hybrid deep learning model that marries InceptionResNetV2 with Bidirectional Long-Short Term Memory (BiLSTM) for advanced deepfake detection. InceptionResNetV2 is a 164-layer deep network initially trained on the ImageNet dataset, combining the strengths of inception architecture with residual networks, efficiently reducing training time and addressing the degradation problem. 

In our model, we've leveraged a customized InceptionResNetV2 to extract visual artifacts from video frames, employing it up to the Inception-ResNet1-C block. We've introduced two dense layers with 128 units and frozen all layers, except the last four, for computational efficiency. The feature vectors are processed through a TimeDistributed layer and then passed through two Bidirectional LSTM layers, one with 128 units and another with 64 units, to capture long-term patterns. Finally, a dense layer with ReLU activation and a fully connected dense layer with softmax activation are employed for frame classification.

### Majority Voting Rule for Video Classification:

In my approach, I've developed a technique to classify individual frames within the video as either genuine or manipulated. To determine the overall category of the video, whether it's real or fake, I've employed a method known as the Majority Voting Rule.

Here's how it works: For each frame in the video, I've collected predictions from my model. I count how many of these predictions indicate that the frame is real and how many suggest it's fake. If the number of predictions favoring the "real" category is higher for a frame, I assign it as a real frame; otherwise, it's classified as a fake frame.

I repeat this process for all the frames in the video, maintaining counts of real and fake frames. At the end of this analysis, I compare the counts and determine the category of the entire video based on the majority. If the number of real frames surpasses the number of fake frames, I classify the video as "REAL"; otherwise, it's labeled as "FAKE."

This Majority Voting Rule enables me to make a confident decision about the authenticity of the video based on the cumulative assessment of its constituent frames. It's a pivotal component of my video classification process, ensuring a robust and reliable approach to deepfake detection.

## Architectural Diagram of my model

![Architecture Diagram](https://github.com/fromjyce/DeepfakeDetection/blob/main/ArchitectureDiagram.png)

## Running of the Programs
The programs have been tested on Google Colab for training and testing, while I used the Visual Studio Code IDE in Windows 11 to develop a website for this model. ***You are free to choose any IDE that suits your needs.***

## Contact
If you come across any mistakes in the programs or have any suggestions for improvement, please feel free to contact me <jaya2004kra@gmail.com>. I appreciate any feedback that can help me improve my coding skills

## License
All the programs in this repository are licensed under the MIT License. You can use them for educational purposes and modify them as per your requirements. ***However, I do not take any responsibility for the accuracy or reliability of the programs.***

## MY SOCIAL PROFILES:
### [LINKEDIN](https://www.linkedin.com/in/jayashrek/)

