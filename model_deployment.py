import cv2
import numpy as np
from tensorflow.keras.models import load_model

img_size = (128, 128)  
model = load_model('drowsiness_detection_model.h5')

def detect_drowsiness(frame, model):
    img = cv2.resize(frame, img_size)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return np.argmax(prediction, axis=1)[0]

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Could not read frame.")
            break
        
        result = detect_drowsiness(frame, model)
        labels = ['Closed', 'No Yawn', 'Open', 'Yawn']
        label = labels[result]
        color = (0, 0, 255) if result in [0, 3] else (0, 255, 0)
        
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Driver Drowsiness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
