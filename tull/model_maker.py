from embeddings_model import *
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import cv2
import joblib

X = []
Y = []

dir = "../../dataset/Knut"
fir = "../../dataset/Vanlige_ansikter_/604ansikter"


def embeddings(image_path, label):
    image = cv2.imread(image_path)
    # Assuming you have a function to obtain the 512-dimensional embeddings
    embedding = get_embeddings(cropp_face(image))
    if embedding is None:
        return
    X.append(embedding.flatten())
    Y.append(label)


# Process the first set of images (label 0)
images = os.listdir(dir)
for i in images:
    image_path = os.path.join(dir, i)
    embeddings(image_path, label=0)

# Process the second set of images (label 1)
images = os.listdir(fir)
for j in images:
    image_path = os.path.join(fir, j)
    embeddings(image_path, label=1)

# Convert lists to NumPy arrays
X = np.array(X)
Y = np.array(Y)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
k = 3  # Choose the number of neighbors (you can adjust this)
knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

model_filename = "knn_face_recognition_model.pkl"
joblib.dump(knn_classifier, model_filename)
