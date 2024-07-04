# Driver Drowsiness Detection

This project aims to detect driver drowsiness using a machine learning model trained on a dataset of images of open and closed eyes, as well as yawning and non-yawning faces.

## Dataset

Download the dataset from the following link:
[Drowsiness Dataset](https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset)

## Setup

1. Clone the repository or download the project files.
2. Ensure you have Python installed on your system.
3. Install the required packages by running:

   ```bash
   pip install -r requirements.txt
   ```

## Running the Project

1. **Preprocess the dataset:**

   Run the preprocessing script to load and preprocess the dataset.

   ```bash
   python preprocessing.py
   ```
2. **Train the model:**

   Run the model training script to train the convolutional neural network (CNN) on the preprocessed dataset.

   ```bash
   python model_training.py
   ```
3. **Deploy the model:**

   Run the model deployment script to use the trained model for real-time drowsiness detection using your webcam.

   ```bash
   python model_deployment.py
   ```

## Files Description

- `preprocessing.py`: Script to load, preprocess the dataset, and split it into training and testing sets.
- `model_training.py`: Script to define, compile, and train the CNN model. The trained model is saved as `drowsiness_detection_model.h5`.
- `model_deployment.py`: Script to deploy the trained model for real-time drowsiness detection using a webcam.

## Requirements

The project requires the following Python packages:

- numpy
- opencv-python
- tensorflow
- scikit-learn

These are listed in the `requirements.txt` file. You can install them using:

```bash
pip install -r requirements.txt
```
