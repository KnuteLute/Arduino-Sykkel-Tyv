ASK=1
KNUT=2
KRISTOFFER=4
SIGURD=3

import os
import sys
import cv2

def load_assets(path):

    folders = ['Ask', 'Knut', 'Jet', 'Sigurd']
    if not (folders <= os.listdir(path)):
        raise Exception('expected different folder')

    dataset = {}
    for folder, id in zip(folders, [ASK, KNUT, KRISTOFFER, SIGURD]):
        for filepath in os.listdir(os.path.join(path, folder)):
            dataset.insert(id, filepath)

    print(dataset)
    return dataset

train_custom_model(dataset):

    model = cv2.dnn.readNetFromTorch('path_to_model.t7')

    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    model.setInput(blob)
    embeddings = model.forward()

def test():
    model = cv2.dnn.readNetFromTorch('path_to_model.t7')

    img = cv2.imread('image.jpg')
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123), swapRB=True)

    model.setInput(blob)
    embeddings = model.forward()
   
)

if __name__ == '__main__':

    args = sys.argv

    if len(args) != 2:
        sys.exit('please provide a path to dataset as command line argument, example: \"python train_model.py dataset\"')
    else:        
        path = sys.argv[1]

    load_assets(path)
    