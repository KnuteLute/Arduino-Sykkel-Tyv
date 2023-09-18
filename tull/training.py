import cv2
import sys
import os
import numpy as np

def train_embedding_img(img, model):
    """Function returns embedding, returns None if face image is too small"""
    img = cv2.imread('image.jpg')
    h, w = img.shape[2:]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    model.setInput(blob)
    embeddings = model.forward()
    if len(embeddings) > 0:
        i = np.argmax(embeddings[0, 0, :, 2])
        confidence = embeddings[0, 0, i, 2]
        # removed The Confidence checker

        box = embeddings[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        face = img[startY:endY, startX:endX]
        fH, fW = face.shape[:2]
        if fW < 20 or fH < 20:
            return None  #

        faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
        model.setInput(faceBlob)
        vec = model.forward()
        return vec.flatten


def train_embedding_person(folder, model, id):

    for img in folder:
        train_embedding_img(img, model)


#def get_final_embeddings(id):