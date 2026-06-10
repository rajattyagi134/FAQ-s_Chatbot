from ollama import embeddings
import numpy as np


def get_embedding(text):

    response = embeddings(
        model="nomic-embed-text",
        prompt=text
    )

    return response["embedding"]


def similarity(vector1, vector2):

    return np.dot(
        vector1,
        vector2
    )