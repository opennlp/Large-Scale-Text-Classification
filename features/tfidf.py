from sklearn.feature_extraction.text import TfidfVectorizer
from constants import commonconstants


class TfIdf:
    def __init__(self, max_feature_num=commonconstants.EMBEDDING_DIMENSION, ngram_range_tuple=(1, 3)):
        self.tf_idf = TfidfVectorizer(max_features=max_feature_num, ngram_range=ngram_range_tuple,input='content')

    def get_feature_set(self, train_data):
        return self.tf_idf.transform(train_data).todense()

    def fit_feature_model(self, train_data):
        self.tf_idf.fit(train_data)

    def get_feature_model(self):
        return self.tf_idf
