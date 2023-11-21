import cv2
import joblib
import numpy as np
from embeddings_model import *
import time
<<<<<<< HEAD
from playsound import playsound

# Load the trained models
model_filename = "tull\knn_face_recognition_model_ny.pkl"
loaded_model = joblib.load(model_filename)
loaded_svm_model = joblib.load("tull\svm_face_model_ny.joblib")
face_cascade = cv2.CascadeClassifier('tull\haarcascade_frontalface_default.xml')
=======


# Load the trained models
model_filename = "knn_face_recognition_model1.pkl"
loaded_model = joblib.load(model_filename)
loaded_svm_model = joblib.load("svm_face_model.joblib")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
>>>>>>> e52165a9050bbf5e432841aba7e77f39723ae250

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Alarm and timing variables
alarm_triggered = False
start_time = time.time()
approved_start_time = None
<<<<<<< HEAD
alarm_deactivated_time = None

# Variables for motion detection
first_frame = None
area_of_interest = (300, 100, 600, 300)  # Define your ROI (x, y, width, height)
=======
>>>>>>> e52165a9050bbf5e432841aba7e77f39723ae250

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

<<<<<<< HEAD
    # Motion Detection Logic
    gray_blur = cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame is None:
        first_frame = gray_blur
        continue

    frame_delta = cv2.absdiff(first_frame, gray_blur)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = thresh[area_of_interest[1]:area_of_interest[1] + area_of_interest[3],
                    area_of_interest[0]:area_of_interest[0] + area_of_interest[2]]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours are found, motion is detected
    if len(contours) > 0:
        if not alarm_triggered:
            start_time = time.time()
            alarm_triggered = True

    face_detected = False
    approved_face_detected = False

    # Face Recognition Logic
=======
    face_detected = False
    approved_face_detected = False

>>>>>>> e52165a9050bbf5e432841aba7e77f39723ae250
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
<<<<<<< HEAD
            if label == 0 and proba > 0.85:
=======
            if label == 0 and proba > 0.9:
>>>>>>> e52165a9050bbf5e432841aba7e77f39723ae250
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

<<<<<<< HEAD
            label_text = "Person " + str(label) + " " + str(round(proba, 4)) + "%"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Check if the alarm should be triggered
    if not approved_face_detected and (time.time() - start_time > 25) and alarm_triggered:
        playsound("tull/alarm_sound.mp3")
        print("Alarm triggered")

    # Check if an approved face has been detected continuously for 3 seconds
    if approved_face_detected and approved_start_time is not None:
        if time.time() - approved_start_time >= 3:
            alarm_triggered = False
            alarm_deactivated_time = time.time()
            print("Alarm deactivated")

    if alarm_deactivated_time is not None and (time.time() - alarm_deactivated_time > 120):
        alarm_triggered = True
        alarm_deactivated_time = None
        print("Alarm activated")

=======
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

>>>>>>> e52165a9050bbf5e432841aba7e77f39723ae250
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()