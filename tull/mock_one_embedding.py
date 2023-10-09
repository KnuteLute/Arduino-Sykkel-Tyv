import numpy as np
import random
from mock_embeddings import mock_embeddings  # Importing mock_embeddings from mock_embeddings.py

# Variable to store the new generated embedding
latest_embedding = None

def generate_similar_mock_embedding():
    global latest_embedding

    # Choose a random person from existing mock embeddings
    random_person = random.choice(list(mock_embeddings.keys()))

    # Choose a random degree of similarity (0 means not similar, 1 means identical)
    similarity = random.uniform(0, 1)

    # If similarity is less than 0.2, generate a completely random embedding
    if similarity < 0.2:
        new_embedding = np.random.uniform(-1, 1, 512)  # 512 dimensions and values between -1 and 1
    else:
        # Choose a random embedding from this person's existing embeddings
        base_embedding = random.choice(mock_embeddings[random_person])

        # Add some noise to the base embedding based on the degree of similarity
        noise = np.random.uniform(-1, 1, 512) * (1 - similarity)  # 512 dimensions and values between -1 and 1

        # Generate the new, similar embedding
        new_embedding = base_embedding + noise

    # Clip values to make sure they are in the range [-1, 1]
    new_embedding = np.clip(new_embedding, -1, 1)

    # Save the new embedding to the variable
    latest_embedding = new_embedding

    return new_embedding

# Example usage:
new_embedding = generate_similar_mock_embedding()
print("Generated similar mock embedding:", new_embedding)