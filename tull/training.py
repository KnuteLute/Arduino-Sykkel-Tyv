import cv2
import sys
import os
import numpy as np
from keras_facenet import FaceNet
import facenet.src.facenet as facenet
embedder = FaceNet()

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


def cropp_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(160, 160))

    # Loop through the detected faces and crop each one
    for (x, y, w, h) in faces:
        # Ensure that the face is at least 160x160 pixels in size
        if w >= 160 and h >= 160:
            cropped_face = img[y:y + h, x:x + w]

    return cropped_face

def get_embeddings(img):
    image = cropp_face(img)
    image = cv2.resize(img, (160, 160))  # Resize to (160, 160) pixels
    # image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add a batch dimension

    # Compute embeddings for the image
    embeddings = embedder.embeddings(image)
    return embeddings