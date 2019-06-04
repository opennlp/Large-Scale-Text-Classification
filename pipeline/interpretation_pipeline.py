from interpretation import instance_explanation
from commonutils import dataframeutils, merge_utils, visualizationutils
import numpy as np
from factory import vectorizer_factory
from preprocessdata import preprocess
from constants import commonconstants
from factory import classifier_factory


def execute_interpretation_pipeline(filepath, vectorizer_name,classifier_name,
                                    max_features_to_show=commonconstants.MAX_FEATURES_TO_SHOW):
    data_filename = filepath.rsplit('\\',1)[1]
    dataframe = dataframeutils.get_data_frame(filepath)
    unique_class_list = np.unique(dataframe['class_label'].values).tolist()
    dataframe['text'] = preprocess.text_clean_pipeline_list(list(dataframe['text'].values))
    train_features = vectorizer_factory.get_vectorized_text(list(dataframe['text'].values), vectorizer_name)
    classifier_list, classifier_name_list = classifier_factory.get_classifier_from_factory(classifier_name)
    vector_transformer = instance_explanation.VectorTransformer(vectorizer_name)
    for classifier, name_classifier in zip(classifier_list, classifier_name_list):
        try:
            classifier.fit(train_features, dataframe['class_label'].values)
            classifier_pipeline = instance_explanation.get_pipeline_for_classification(vector_transformer, classifier)
            random_index_list = dataframeutils.get_random_integer_list(0, len(dataframe), commonconstants.MAX_INTERPRETATIONS_TO_GENERATE)
            for text_index in random_index_list:
                file_to_save_interpretation = data_filename.replace('.csv','') + '_' + vectorizer_name \
                                              + '_' + name_classifier + '_' + str(text_index) + '.html'
                file_to_save_interpretation = commonconstants.INTERPRETATION_FOLDER_PATH + file_to_save_interpretation
                print(instance_explanation.get_explanation_for_instance((dataframe['text'].values)[text_index], classifier_pipeline,
                                                                  unique_class_list,max_num_features_to_show=max_features_to_show,
                                                                  file_to_save=file_to_save_interpretation))
        except Exception as e:
            print(e)


def execute_interpretation_pipeline_for_stacked_bars(filepath, vectorizer_name_list,
                                                     classifier_name,
                                                     max_features_to_show=commonconstants.MAX_FEATURES_TO_SHOW):
    data_filename = filepath.rsplit('\\',1)[1]
    dataframe = dataframeutils.get_data_frame(filepath)
    unique_class_list = np.unique(dataframe['class_label'].values).tolist()
    dataframe['text'] = preprocess.text_clean_pipeline_list(list(dataframe['text'].values))
    dict_weight_list = list([])

    for vectorizer_name in vectorizer_name_list:
        train_features = vectorizer_factory.get_vectorized_text(list(dataframe['text'].values), vectorizer_name)
        classifier_list, classifier_name_list = classifier_factory.get_classifier_from_factory(classifier_name)
        vector_transformer = instance_explanation.VectorTransformer(vectorizer_name)
        for classifier, name_classifier in zip(classifier_list, classifier_name_list):
            try:
                classifier.fit(train_features, dataframe['class_label'].values)
                classifier_pipeline = instance_explanation.get_pipeline_for_classification(vector_transformer, classifier)
                random_index_list = [4645]
                for text_index in random_index_list:
                    file_to_save_interpretation = data_filename.replace('.csv','') + '_' + vectorizer_name \
                                                  + '_' + name_classifier + '_' + str(text_index) + '.html'
                    file_to_save_interpretation = commonconstants.INTERPRETATION_FOLDER_PATH + file_to_save_interpretation
                    weight_tuple_list = instance_explanation.get_explanation_for_instance((dataframe['text'].values)[text_index], classifier_pipeline,
                                                                      unique_class_list,max_num_features_to_show=max_features_to_show,
                                                                      file_to_save=file_to_save_interpretation)
                    dict_weight_list.append(merge_utils.convert_word_weight_tuple_list_to_dict(weight_tuple_list,vectorizer_name))
            except Exception as e:
                print(e)
    final_dataframe = merge_utils.get_complete_dataframe_from_dict(dict_weight_list)
    visualizationutils.plot_stacked_barchart(x_name='weights', y_name='words',
                                             dataframe=final_dataframe,groupby='vectorizer')
