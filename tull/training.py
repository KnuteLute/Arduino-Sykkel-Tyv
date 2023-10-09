import cv2
import sys
import os
import numpy as np
import mxnet as mx
import onnx
from mxnet.contrib.onnx.onnx2mx.import_model import import_model
import openface
from keras_facenet import FaceNet


def get_embbedings(model, img):
    face_pixles = img.astype('float32')
    mean, std = face_pixles.mean(), face_pixles.std()
    face_pixles = (face_pixles - mean) / std
    samples = np.expand_dims(face_pixles, axis=0)
    yhat = model.predict(samples)
    return yhat[0]


def train_embedding_img(imgPath, detector, embedder, confidence_threshold=0.8):
    """
    Function takes in image path name. The detection model (for face), and embedder model.
    Function returns one 128-d array of embedding, returns None if the face image is too small, or it does not recognize the image
    """
    img = cv2.imread(imgPath)

    # Resize the image
    img = cv2.resize(img, (600, 600))  # Adjust the size accordingly

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    detector.setInput(blob)
    detections = detector.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            """input_blob = np.expand_dims(detections, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)

            # Normalize embedding obtained from forward pass to a unit vector
            embedding = model.get_outputs()[0].squeeze()
            embedding /= embedding.norm()"""
            return get_embbedings(embedder, detections)
        else:
            return None


def train_embedding_person(folder, model, id):
    for img in folder:
        train_embedding_img(img, model)


# def get_final_embeddings(id):

detector = cv2.dnn.readNetFromCaffe('../models/deploy.prototxt.txt',
                                    '../models/res10_300x300_ssd_iter_140000.caffemodel')

# embedder = cv2.dnn.readNetFromTorch('../models/nn4.v1.t7')

# model = openface.TorchNeuralNet('../models/nn4.v1.t7', imgDim=96, cuda=False)

embedder = FaceNet()

train_embedding_img('1.jpg', detector, embedder)
