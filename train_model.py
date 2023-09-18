ASK=1
KNUT=2
JET=4
SIGURD=3

import os
import sys
import cv2

def load_assets(path):

    folders = ['Ask', 'Knut', 'Jet', 'Sigurd']
    if not (folders <= os.listdir(path)):
        raise Exception('expected different folder')

    dataset = {}
    for folder, id in zip(folders, [ASK, KNUT, JET, SIGURD]):
        for filepath in os.listdir(os.path.join(path, folder)):
            dataset.insert(id, filepath)

    print(dataset)
    return dataset

def train_embedding_img(img, model):



    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    model.setInput(blob)
    embeddings = model.forward()


def train_embedding_person(folder, model, id):

    for img in folder:
        train_embedding_img(img, model)


#def get_final_embeddings(id):



if __name__ == '__main__':

    args = sys.argv

    if len(args) != 2:
        sys.exit('please provide a path to dataset as command line argument, example: \"python train_model.py dataset\"')
    else:        
        path = sys.argv[1]

    datasetfiles = load_assets(path)

    model = cv2.dnn.readNetFromTorch('models.t7')

    train_embedding_person(datasetfiles[ASK], model, ASK)
    train_embedding_person(datasetfiles[KNUT], model, KNUT)
    train_embedding_person(datasetfiles[JET], model, JET)
    train_embedding_person(datasetfiles[SIGURD], model, SIGURD)

# går det greit at dette er min gren


