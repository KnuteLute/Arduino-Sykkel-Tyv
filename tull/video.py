import cv2
import joblib
import numpy as np
from embeddings_model import *
import time


# Load the trained models
model_filename = "knn_face_recognition_model1.pkl"
loaded_model = joblib.load(model_filename)
loaded_svm_model = joblib.load("svm_face_model.joblib")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Alarm and timing variables
alarm_triggered = False
start_time = time.time()
approved_start_time = None

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

    face_detected = False
    approved_face_detected = False

    for (x, y, w, h) in faces:
        face_detected = True
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))
        embedding = get_embeddings(face)

        if embedding is not None:
            flattened_embedding = embedding.flatten()
            preds = loaded_svm_model.predict_proba([flattened_embedding])[0]
            j = np.argmax(preds)
            proba = preds[j]
            label = loaded_svm_model.predict([flattened_embedding])[0]
            if label == 0 and proba > 0.9:
                label = loaded_model.predict([flattened_embedding])[0]
                if label in ["Jet", "Sigurd", "Knut", "Ask"]:
                    if approved_start_time is None:
                        approved_start_time = time.time()
                    approved_face_detected = True
                else:
                    approved_start_time = None
            else:
                label = "Vanlig Ansikt"
                approved_start_time = None

            label_text = "Person " + str(label) + " " + str(round(proba,4)) + "%"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Check if the alarm should be triggered
    if not approved_face_detected and (time.time() - start_time > 10) and not alarm_triggered:
        alarm_triggered = True


    # Check if an approved face has been detected continuously for 3 seconds
    if approved_face_detected and approved_start_time is not None:
        if time.time() - approved_start_time >= 3:
            alarm_triggered = False
            break

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()