from features import embedding
from features import hashfeatures
from features import tfidf
from constants import commonconstants


def get_vectorized_text(sentence_list, vectorizer_name):
    if vectorizer_name.lower() == commonconstants.FEATURE_HASH:
        return hashfeatures.FeatureHash().get_feature_set(sentence_list)
    elif vectorizer_name.lower() == commonconstants.TF_IDF:
        tf_idf = tfidf.TfIdf()
        tf_idf.fit_feature_model(sentence_list)
        return tf_idf.get_feature_set(sentence_list)
    elif vectorizer_name.lower() in commonconstants.NEURAL_EMBEDDING_LIST:
        return embedding.get_word_embeddings(sentence_list, model_to_use=vectorizer_name)
    else:
        raise Exception("Vectorizer either unsupported or a mis-spelled  %s" % vectorizer_name)
