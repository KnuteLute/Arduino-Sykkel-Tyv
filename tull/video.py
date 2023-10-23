import cv2
import joblib
import numpy as np
from embeddings_model import *

# Load the trained KNeighborsClassifier model
model_filename = "knn_face_recognition_model1.pkl"
loaded_model = joblib.load(model_filename)
loaded_svm_model = joblib.load("svm_face_model.joblib")

# Load a face detection classifier (e.g., Haar Cascade or a deep learning model)
# You'll need to download and load a pre-trained face detection model here
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam capture
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam, or adjust as needed

while True:
    ret, frame = cap.read()  # Read a frame from the webcam
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection on the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region from the frame
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (160, 160))

        # Obtain the face embedding using your get_embeddings function
        embedding = get_embeddings(face)
        if embedding is not None:
            # Flatten the embedding
            flattened_embedding = embedding.flatten()

            label = loaded_svm_model.predict([flattened_embedding])[0]
            if label == 0:
                # Use the loaded KNeighborsClassifier model to predict the label
                label = loaded_model.predict([flattened_embedding])[0]
            else:
                label = "Vanlig Ansikt"
            # decision = loaded_model.decision_function([flattened_embedding])

            # Calculate the confidence (normalized distance from the decision boundary)
            # confidence = 1 / (1 + np.exp(-decision))

            # Display the label and confidence on the frame
            label_text = "Person " + str(label)
            # accuracy_text = "Confidence: {:.2f}%".format(confidence[0] * 100)

            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # cv2.putText(frame, accuracy_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with face recognition results
    cv2.imshow('Face Recognition', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
