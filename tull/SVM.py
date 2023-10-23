from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from embeddings_model import *
import os
from joblib import dump

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
    embeddings(image_path, label=0)

images = os.listdir(rir)
for i in images:
    image_path = os.path.join(rir, i)
    embeddings(image_path, label=0)

images = os.listdir(eir)
for i in images:
    image_path = os.path.join(eir, i)
    embeddings(image_path, label=1)


# Process the second set of images (label 1)
images = os.listdir(fir)
for j in images:
    image_path = os.path.join(fir, j)
    embeddings(image_path, label=1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# Create and train SVM classifier
clf = SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
dump(clf, 'svm_face_model.joblib')