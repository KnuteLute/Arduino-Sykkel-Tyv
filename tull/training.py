import cv2
import sys
import os
import numpy as np


def train_embedding_img(imgPath, detector, embedder, confidence_threshold=0.8):
    """
    Function takes in image path name. The detection model (for face), and embedder model.\n
    Function returns one 128-d array of embedding , returns None if face image is too small, or it does not
    recognize the image
    """
    img = cv2.imread(imgPath)
    img = imgPath.resize(img, width=600)
    h, w = imgPath.shape[:2]
    blob = cv2.dnn.blobFromImage(imgPath, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    detector.setInput(blob)
    embeddings = detector.forward()
    if len(embeddings) > 0:
        i = np.argmax(embeddings[0, 0, :, 2])
        confidence = embeddings[0, 0, i, 2]

        if confidence > confidence_threshold:
            box = embeddings[0, 0, i, 3:7] * np.array([w, h, w, h])
            startX, startY, endX, endY = box.astype("int")
            face = img[startY:endY, startX:endX]
            fH, fW = face.shape[:2]
            if fW < 20 or fH < 20:
                return None

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            return vec.flatten()
        else:
            return None


def train_embedding_person(folder, model, id):

    for img in folder:
        train_embedding_img(img, model)

# def get_final_embeddings(id):


model = cv2.dnn.readNetFromTorch('../models/nn4.small2.v1.t7')
detector = cv2.dnn.readNetFromCaffe('../models/deploy.prototxt.txt', '../res10_300x300_ssd_iter_140000.caffemodel')