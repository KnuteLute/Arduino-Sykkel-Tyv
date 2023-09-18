import cv2
import os
from tull import classification_model, file, recognize
from tull.utils import ASK, KNUT, JET, SIGURD, ANDRE

if __name__ == '__main__':

    print('loading embedding model')
    embeddings_model = cv2.dnn.readNetFromTorch(os.path.join('models', 'nn4.small2.v1.t7'))

    print('loading classification models')
    name_weights = {
        ASK:file.load_weights('ask'),
        KNUT:file.load_weights('knut'),
        JET:file.load_weights('jet'), 
        SIGRUD:file.load_weights('sigurd'),
    }

    name_models = {}
    for name, weight in name_weights:
        classification_model = classification_model.ClassificationModel()
        classification_model.set_weights(weight)
        name_models[name] = classification_model
 
    print('running facial recognition')
    recognize.run(embeddings_model, name_models)