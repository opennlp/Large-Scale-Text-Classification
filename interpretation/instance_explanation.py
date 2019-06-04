from factory import vectorizer_factory
from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer


class VectorTransformer(TransformerMixin):
    def __init__(self, vectorizer_name):
        self.vectorizer_name = vectorizer_name

    def fit(self,X, y=None):
        pass

    def transform(self, sentence_list, y=None):
        return vectorizer_factory.get_vectorized_text(sentence_list,self.vectorizer_name)


def get_pipeline_for_classification(feature_transformer, trained_model):
    return make_pipeline(feature_transformer, trained_model)


def get_explanation_for_instance(text_string,classifier_function, class_list, max_num_features_to_show=10, file_to_save='explain.html'):
    explainer = LimeTextExplainer(class_names=class_list,random_state=42)
    explained_instance = explainer.explain_instance(text_string, classifier_function.predict_proba,
                                                    num_features=max_num_features_to_show, top_labels=len(class_list))
    explained_instance.save_to_file(file_to_save)
    return explained_instance.as_list()
