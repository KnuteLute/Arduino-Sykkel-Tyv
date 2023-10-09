from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from mock_embeddings import mock_embeddings  # Import mock embeddings


class ClassificationModel:
    def __init__(self):
        # Initialize KNN
        self.knn = KNeighborsClassifier(n_neighbors=3)

        # Prepare data for KNN
        X = []
        y = []
        for label, embeddings in mock_embeddings.items():
            for emb in embeddings:
                X.append(emb)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        # Fit KNN
        self.knn.fit(X, y)

    def forward(self, embedding_positive, embedding_negative):
        assert self.weights != None
        assert self.anchor_embedding != None
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self):
        raise NotImplementedError()

    def set_anchor_embedding(self):
        raise NotImplementedError()

    def match(self, embedding):
        # Predict the label of the new embedding
        prediction = self.knn.predict([embedding])

        # Get distances to all neighbors
        distances, _ = self.knn.kneighbors([embedding])

        # If distance to the nearest neighbor is above a threshold, it's an unknown face
        if np.min(distances) > 1:  # Changed threshold to match new value range
            print("Alert! Unknown face detected.")
            return 0.0
        else:
            print(f"Face recognized as {prediction[0]}.")
            return 1.0  # Or some other measure of match quality


