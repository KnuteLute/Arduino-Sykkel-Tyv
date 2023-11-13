from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from embeddings_model import *
import os
from joblib import dump
from sklearn.preprocessing import LabelEncoder

X = []
Y = []

dir = "../../dataset/Knut"
pir = "../../dataset/Ask"
rir = "../../dataset/Sigurd"
eir = "../../dataset/Jet"

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

images = os.listdir(pir)
for i in images:
    image_path = os.path.join(pir, i)
    embeddings(image_path, label=1)

images = os.listdir(rir)
for i in images:
    image_path = os.path.join(rir, i)
    embeddings(image_path, label=2)

images = os.listdir(eir)
for i in images:
    image_path = os.path.join(eir, i)
    embeddings(image_path, label=3)


# Process the second set of images (label 1)
images = os.listdir(fir)
for j in images:
    image_path = os.path.join(fir, j)
    embeddings(image_path, label=4)

recognizer = SVC(C=1.0, kernel="rbf", probability=True)
recognizer.fit(X, Y)
# Predict on the test set
dump(recognizer, 'svm_face_model.joblib')