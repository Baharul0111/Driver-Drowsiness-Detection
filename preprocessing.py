import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

dataset_path = '/Users/baharulislam/Desktop/MACHINE_LEARNING/train'
img_size = (128, 128)  

def load_and_preprocess(data_path, img_size):
    images = []
    labels = []
    label_map = {'Closed': 0, 'no_yawn': 1, 'Open': 2, 'yawn': 3}

    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if not os.path.isdir(label_path):
            continue
        for img_file in os.listdir(label_path):
            if img_file.startswith('.'):
                continue
            img_path = os.path.join(label_path, img_file)
            image = cv2.imread(img_path)
            if image is not None:
                image = cv2.resize(image, img_size)
                images.append(image)
                labels.append(label_map[label])
                
    images = np.array(images)
    labels = np.array(labels)
   
    images = images / 255.0
    return images, to_categorical(labels, num_classes=4)

images, labels = load_and_preprocess(dataset_path, img_size)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
