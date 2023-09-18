import cv2
import sys
import os
import numpy as np

def train_embedding_img(img, model):

    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    model.setInput(blob)
    embeddings = model.forward()


def train_embedding_person(folder, model, id):

    for img in folder:
        train_embedding_img(img, model)


#def get_final_embeddings(id):

