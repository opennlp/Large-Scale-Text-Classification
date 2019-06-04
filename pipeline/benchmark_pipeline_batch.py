from loaddata import local_load
from commonutils import metadatautils, expectationutils, batchutils
from commonutils import dataframeutils, merge_utils, dbutils, logutils
from preprocessdata import preprocess
from factory import vectorizer_factory, classifier_factory
from splitdata import create_train_test_split
from evaluation import classification_metrics, metric_helpers
import numpy as np
from constants import commonconstants

LOGGER = logutils.get_logger("Benchmark Pipeline")


class BenchmarkBatch:
    def __init__(self,category='all',data_size=50000,vectorizer='featurehash',model_type='non_neural',persist_type=None):
        self.data_category = category
        self.data_size = data_size
        self.vectorizer = vectorizer
        self.model_type = model_type
        self.persist_type = persist_type

    def load_data_file_path_dict(self):
        category_filepath_dict = local_load.get_file_paths_for_categories(self.data_category)
        return category_filepath_dict

    def get_data_frame(self,filepath):
        df = dataframeutils.get_data_frame(filepath)
        df.dropna(inplace=True)
        return df

    def validate_expectation(self,dataframe):
        return expectationutils.check_num_row_in_dataframe(dataframe,self.data_size)

    @staticmethod
    def preprocess_and_clean_text(dataframe):
        dataframe['text'] = preprocess.text_clean_pipeline_list(list(dataframe['text'].values))
        return dataframe

    def get_data_metadata_summary(self,dataframe):
        metadata_summary_dict = metadatautils.get_text_summary_statistics(list(dataframe['text'].values))
        metadata_summary_dict['data_category'] = self.data_category
        metadata_summary_dict['vectorizer'] = self.vectorizer
        metadata_summary_dict['imbalance_measure'] = metadatautils.get_class_imbalance_score(dataframe['class_label'].values)
        metadata_summary_dict['num_rows'] = len(dataframe)
        return metadata_summary_dict

    def get_vectorized_features(self,dataframe):
        return vectorizer_factory.get_vectorized_text(list(dataframe['text'].values), vectorizer_name=self.vectorizer)

    @staticmethod
    def get_train_test_split(data_values,data_labels):
        return create_train_test_split.get_train_test_data(data_values,data_labels)

    def get_prediction_models(self):
        return classifier_factory.get_classifier_from_factory(self.model_type)

    @staticmethod
    def fit_prediction_model(prediction_model, X_train, y_train):
        prediction_model.fit(X_train,y_train)

    @staticmethod
    def get_model_predictions(trained_prediction_model, X_test):
        return trained_prediction_model.predict(X_test)

    @staticmethod
    def get_classification_metrics_summary(trained_model, gold_labels, prediction_labels):
        summary_metric_dict = dict({})
        accuracy_score = classification_metrics.get_accuracy_score(gold_labels, prediction_labels)
        precision_array, recall_array, f1_array, support_array = classification_metrics.get_precision_recall_f1_support(gold_labels, prediction_labels)
        class_iterable = trained_model.classes_
        precision_dict = metric_helpers.get_class_specific_metrics(class_iterable, precision_array)
        recall_dict = metric_helpers.get_class_specific_metrics(class_iterable, recall_array)
        f1_dict = metric_helpers.get_class_specific_metrics(class_iterable, f1_array)
        support_dict = metric_helpers.get_class_specific_metrics(class_iterable, support_array)
        summary_metric_dict['accuracy'] = accuracy_score
        summary_metric_dict['class_specific_support'] = support_dict
        summary_metric_dict['class_specific_precision'] = precision_dict
        summary_metric_dict['class_specific_recall'] = recall_dict
        summary_metric_dict['class_specific_f1'] = f1_dict
        summary_metric_dict['macro_precision'] = metric_helpers.get_macro_metrics(precision_array)
        summary_metric_dict['macro_recall'] = metric_helpers.get_macro_metrics(recall_array)
        summary_metric_dict['macro_f1'] = metric_helpers.get_macro_metrics(f1_array)
        summary_metric_dict['macro_support'] = metric_helpers.get_macro_metrics(support_array)
        return summary_metric_dict

    def persist_results(self, data_to_insert):
        dbutils.check_and_insert_in_db(data_to_insert)

    def execute_pipeline(self):
        category_filepath_dict = self.load_data_file_path_dict()
        LOGGER.info("Loaded Data file path locations for %s " % self.data_category)
        for category_name, filepath_list in category_filepath_dict.items():
            for filepath in filepath_list:
                try:
                    dataset_file_name = str(filepath).rsplit("\\",1)[1]
                    dataframe = self.get_data_frame(filepath)
                    LOGGER.info("Loaded Dataframe for file %s" % dataset_file_name)
                    if self.validate_expectation(dataframe):
                        dataframe = self.preprocess_and_clean_text(dataframe)
                        dataframe = dataframe.sample(frac=1).reset_index(drop=True)
                        print(dataset_file_name)
                        metadata_summary_dict = self.get_data_metadata_summary(dataframe)
                        metadata_summary_dict['data_filename'] = dataset_file_name
                        metadata_summary_dict['category_folder_name'] = category_name
                        unique_classes = np.unique(dataframe['class_label'].values)
                        metadata_summary_dict['num_class_labels'] = len(unique_classes)
                        LOGGER.info("Metadata summary initialized for data %s" % dataset_file_name)
                        test_size, test_start_index = batchutils.get_test_data_size_and_index(len(dataframe),commonconstants.TEST_FRACTION)
                        num_chunks = batchutils.get_number_of_chunks(len(dataframe),commonconstants.CHUNK_SIZE,test_size)
                        start_index = 0
                        classifier_list, classifier_name_list = self.get_prediction_models()

                        for classifier, classifier_name in zip(classifier_list, classifier_name_list):
                            metadata_summary_dict['classifier_name'] = classifier_name
                            print("%s %s %s" %(classifier_name, self.vectorizer, dataset_file_name))
                            for chunk in range(num_chunks):
                                batch_dataframe = dataframe[start_index:start_index+commonconstants.CHUNK_SIZE]
                                train_features = self.get_vectorized_features(batch_dataframe)
                                class_labels = list(batch_dataframe['class_label'].values)
                                if self.model_type.lower() == commonconstants.WARM_START_TYPE:
                                    classifier.fit(train_features, class_labels)
                                else:
                                    classifier.partial_fit(train_features, class_labels, classes=unique_classes)
                                start_index = start_index + commonconstants.CHUNK_SIZE

                            test_frame = dataframe[test_start_index:]
                            X_test = self.get_vectorized_features(test_frame)
                            print(X_test.shape)
                            y_test = list(test_frame['class_label'].values)
                            predicted_labels = self.get_model_predictions(classifier, X_test)
                            metric_summary_dict = self.get_classification_metrics_summary(classifier,y_test,predicted_labels)
                            LOGGER.info("Data classified using featurizer %s and classifier %s" % (self.vectorizer, classifier_name))
                            final_merged_dict = merge_utils.merge_dicts(metadata_summary_dict, metric_summary_dict)
                            print(final_merged_dict)
                            self.persist_results(final_merged_dict)
                            LOGGER.info("Data Inserted for %s" % dataset_file_name)

                except Exception as e:
                    print(e.__str__())
                    LOGGER.error("An error occurred while inserting data %s" % e.__str__())


if __name__ == '__main__':
    for category_name in commonconstants.CATEGORY_LIST:
        for vectorizer_name in commonconstants.VECTORIZER_LIST:
            try:
                b = BenchmarkBatch(category=category_name, vectorizer=vectorizer_name, data_size=50000, model_type='warm')
                b.execute_pipeline()
            except Exception as e:
                 print("In main %s" % e.__str__())
