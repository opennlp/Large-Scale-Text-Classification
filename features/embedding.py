from pymagnitude import *
from pathlib import Path
from constants import modelconstants
import numpy as np
from features import universal_sentence_encoder
from features import flair_embedding


def get_model_path(model_folder_name, model_name):
    current_path = Path(Path.cwd())
    parent_path = current_path.parent
    return parent_path.joinpath(model_folder_name, model_name)


def get_word_embedding_model_object(model_to_use='glove'):
    model_path = get_model_path(modelconstants.MODEL_FOLDER_NAME, modelconstants.GLOVE_MODEL_NAME)
    if model_to_use.lower() == 'word2vec':
        model_path = get_model_path(modelconstants.MODEL_FOLDER_NAME, modelconstants.WORD2VEC_MODEL_NAME)
    if model_to_use.lower() == 'fasttext':
        model_path = get_model_path(modelconstants.MODEL_FOLDER_NAME, modelconstants.FASTTEXT_MODEL_NAME)
    if model_to_use.lower() == 'elmo':
        model_path = get_model_path(modelconstants.MODEL_FOLDER_NAME, modelconstants.ELMO_MODEL_NAME)
    magnitude_vector_object = Magnitude(model_path, case_insensitive=True, ngram_oov=True)
    return magnitude_vector_object


def get_word_embeddings(sentence_list, model_to_use='glove'):
    if model_to_use =='universalencoder':
        sentence_list_actual = [sentence for sentence in sentence_list]
        return universal_sentence_encoder.get_sentence_encoding(sentence_list_actual)
    elif model_to_use.lower() == 'flair':
        return flair_embedding.get_flair_embeddings(sentence_list)
    word_embedding_list = []
    magnitude_vector_object = get_word_embedding_model_object(model_to_use)
    for sentence in sentence_list:
        word_list = sentence.split(' ')
        composite_vector = [0.0 for _ in range(magnitude_vector_object.dim)]
        for word in word_list:
            word_vector = magnitude_vector_object.query(word)
            composite_vector = (np.array(word_vector) + np.array(composite_vector)) / 2.0
        word_embedding_list.append(composite_vector)
    return np.array(word_embedding_list)

