from embeddings_model import get_embeddings
import cv2

img = cv2.imread("1.jpg")
embeddings = get_embeddings(img)
file_path = "../models/baseface/basefaceKNUT.txt"
with open(file_path, "w") as file:
    for embedding in embeddings:
        file.write(str(embedding) + "\n")