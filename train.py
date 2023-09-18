import sys
import argparse
import os
import cv2
import random
from tull import embeddings_model, file, utils, classification_model
from tull.utils import ASK, KNUT, JET, SIGURD, ANDRE

if __name__ == '__main__':

    # force command line arguments
    parser = argparse.ArgumentParser(description='this script train a model for recognising the faces of the legendary Ask, Knut, Sigurd and Jet')
    parser.add_argument('-d', '--data', required=True, help='provide file path to training datset containing folders Ask, Knut, Sigurd and Jet, containing corresponding images')

    args = parser.parse_args()

    # run program
    print('reading dataset names')
    name_pictures = file.load_dataset(args.data)

    name_embeddings = {
        ASK:[],
        KNUT:[],
        JET:[], 
        SIGURD:[],
        ANDRE:[],
    }

    print('loading embedding model')
    embeddings_model = cv2.dnn.readNetFromTorch(os.path.join('models', 'nn4.small2.v1.t7'))

    print('retrieving embeddings from model')
    for name, pics in name_pictures:
        for pic in pics:
            name_embeddings[name].append(embeddings_model.get_embeddings(pic, embeddings_model))

    print('training classification models')
    name_weights = {
        ASK:[],
        KNUT:[],
        JET:[], 
        SIGURD:[],
    }
    for name, embeddings in name_embeddings:

        classification_model = classification_model.ClassificationModel()
        classification_model.setAnchor(random.choice(name_embeddings[name]))

        for embedding in embeddings:
            classification_model.forward(embedding, random.choice(name_embeddings[ANDRE]))

        name_weights[name].append(classification_model.get_weights())
           
    print('saving classification models')
    file.save_weights(name_weights[ASK], 'ask')
    file.save_weights(name_weights[KNUT], 'knut')
    file.save_weights(name_weights[JET], 'jet')
    file.save_weights(name_weights[SIGURD], 'sigurd')

    print('training done')
