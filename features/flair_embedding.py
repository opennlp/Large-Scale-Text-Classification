import numpy as np
from constants import modelconstants
from flair.embeddings import FlairEmbeddings
from flair.data import Sentence


def get_flair_embeddings(sentence_list):
    flair_sentence_object_list = []
    for sentence_string in sentence_list:
        sentence_string = str(sentence_string) + " ."
        flair_sentence_object_list.append(Sentence(sentence_string))
    numpy_embedding_list = list([])
    flair_embedding_forward = FlairEmbeddings(modelconstants.FLAIR_MODEL_NAME)
    for sentence_object in flair_sentence_object_list:
        flair_embedding_forward.embed(sentence_object)
        composite_vector = [0.0 for _ in range(flair_embedding_forward.embedding_length)]
        for token in sentence_object:
            token_embedding = token.embedding.numpy()
            composite_vector = (np.array(composite_vector) + np.array(token_embedding))/2.0
        numpy_embedding_list.append(composite_vector)
    return np.array(numpy_embedding_list)
