import numpy as np

# Dictionary to hold mock embeddings as a hash-map
mock_embeddings = {}

def generate_mock_embeddings(num_people=4, num_embeddings=20, dim=512):
    global mock_embeddings

    for i in range(num_people):
        # Generate 20 random 512-dimensional vectors for each person
        # Values will be between -1 and 1
        person_embeddings = np.random.uniform(-1, 1, (num_embeddings, dim))

        # Save the embeddings for this person in the hash-map
        mock_embeddings[f'person_{i + 1}'] = person_embeddings

# Generate the mock embeddings
generate_mock_embeddings()
