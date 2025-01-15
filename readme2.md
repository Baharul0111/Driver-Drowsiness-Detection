# Driver Drowsiness Detection System

## Overview

This project presents a cutting-edge driver drowsiness detection system that leverages both Large Language Models (LLMs) and deep learning techniques. By analyzing behavioral cues such as facial expressions and eye movements, the system accurately measures driver drowsiness, enhancing road safety and reducing the risk of accidents caused by fatigue.

## Methodology

Our approach integrates the strengths of deep learning for feature extraction with the contextual understanding capabilities of LLMs. The system utilizes the NVIDIA NV-DINOv2 API to extract high-dimensional image embeddings from driver images, capturing intricate details of facial expressions and eye movements. These embeddings are then processed by a neural network to classify the driver's state into one of four categories: Closed, No Yawn, Open, or Yawn.

Unlike traditional methods that rely on limited data types and simpler algorithms, our system employs a more comprehensive data analysis pipeline. This integration addresses the variability of driving conditions and diverse fatigue indicators, resulting in improved accuracy and reliability in real-world scenarios.

### Key Components

- **Data Preparation:** Images are categorized and split into training, validation, and testing datasets. Each image is processed to extract embeddings using the NVIDIA NV-DINOv2 API.
- **Neural Network Architecture:** A simple yet effective neural network with layer normalization, fully connected layers, ReLU activation, dropout, and softmax for classification.
- **Training and Evaluation:** The model is trained using cross-entropy loss and optimized with the Adam optimizer. Performance is evaluated based on accuracy, precision, recall, and F1 score.

## Results

In this study, we propose a new method for driver drowsiness detection combining LLM and deep learning techniques. Our approach utilizes behavioral information, including facial expression and eye movement, to precisely measure drowsiness.

Unlike existing approaches that tend to rely on limited types of data and simple algorithms, our system utilizes the strengths of deep learning for feature extraction coupled with the contextual understanding of LLM. This integration improves accuracy and reliability where previous methods suffer from variability of driving conditions and fatigue indicators.

**Performance Metrics:**

- **Accuracy:** 99.25%
- **Average Precision:** 99%
- **Average Recall:** 99%
- **Average F1 Score:** 99%

These results demonstrate the effectiveness and robustness of our system in accurately detecting driver drowsiness under various conditions.
