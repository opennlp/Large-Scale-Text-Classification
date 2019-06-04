from commonutils import dataframeutils
from constants import commonconstants


def get_top_vectorizer_dict_given_data_classifier_metric(dataframe, data_filename, classifier_name, metric):
    file_filtered_frame = dataframe[(dataframe['data_filename'] == data_filename) & (dataframe['classifier_name'] == classifier_name)]
    if len(file_filtered_frame) > 0:
        file_filtered_frame.reset_index(drop=True, inplace=True)
        filtered_dict_list = file_filtered_frame.to_dict(orient='records')
        metric_dict = dict({})
        for filtered_dict in filtered_dict_list:
            metric_dict[filtered_dict['vectorizer']] = filtered_dict[metric]
        sorted_by_value = sorted(metric_dict.items(), key=lambda value: -value[1])
        return sorted_by_value
    return []


def get_top_classifier_dict_given_data_vectorizer_metric(dataframe, data_filename, vectorizer, metric):
    file_filtered_frame = dataframe[(dataframe['data_filename'] == data_filename) & (dataframe['vectorizer'] == vectorizer)]
    if len(file_filtered_frame) > 0:
        file_filtered_frame.reset_index(drop=True, inplace=True)
        filtered_dict_list = file_filtered_frame.to_dict(orient='records')
        metric_dict = dict({})
        for filtered_dict in filtered_dict_list:
            metric_dict[filtered_dict['classifier_name']] = filtered_dict[metric]
        sorted_by_value = sorted(metric_dict.items(), key=lambda value: -value[1])
        return sorted_by_value
    return []


def get_top_vectorizer_dict(dataframe, metric_name='accuracy', top_n=1, classifier=None):
    data_filename_list = dataframeutils.get_unique_values_from_dataframe(dataframe,'data_filename')
    vectorizer_margin_dict = dict({})
    for vectorizer in commonconstants.VECTORIZER_LIST:
        vectorizer_margin_dict[vectorizer] = 0.0

    if classifier is None:
        classifier_name_list = dataframeutils.get_unique_values_from_dataframe(dataframe, 'classifier_name')
    else:
        classifier_name_list = [classifier]
    top_vectorizer_dict = dict({})
    for data_filename in data_filename_list:
        for classifier_name in classifier_name_list:
            sorted_metric_list = get_top_vectorizer_dict_given_data_classifier_metric(dataframe,
                                                                                      data_filename,
                                                                                      classifier_name,
                                                                                      metric_name)

            if len(sorted_metric_list) >= 2:
                margin_value = get_margin_value(sorted_metric_list)
                vectorizer_margin_dict[sorted_metric_list[0][0]] = vectorizer_margin_dict[sorted_metric_list[0][0]] + margin_value
            sorted_metric_list = sorted_metric_list[:top_n]
            for sorted_metric in sorted_metric_list:
                if sorted_metric[0] in top_vectorizer_dict.keys():
                    top_vectorizer_dict[sorted_metric[0]] = top_vectorizer_dict[sorted_metric[0]] + 1
                else:
                    top_vectorizer_dict[sorted_metric[0]] = 1
    return top_vectorizer_dict, vectorizer_margin_dict


def get_margin_value(sorted_metric_list):
    return abs(sorted_metric_list[0][1] - sorted_metric_list[1][1])


def get_top_classifier_dict(dataframe, metric_name='accuracy', top_n=1, vectorizer=None):
    data_filename_list = dataframeutils.get_unique_values_from_dataframe(dataframe,'data_filename')
    if vectorizer is None:
        vectorizer_name_list = dataframeutils.get_unique_values_from_dataframe(dataframe, 'vectorizer')
    else:
        vectorizer_name_list = [vectorizer]
    top_classifier_dict = dict({})
    for data_filename in data_filename_list:
        for vectorizer_name in vectorizer_name_list:
            sorted_metric_list = get_top_classifier_dict_given_data_vectorizer_metric(dataframe,
                                                                                      data_filename,
                                                                                      vectorizer_name,
                                                                                      metric_name)
            sorted_metric_list = sorted_metric_list[:top_n]
            for sorted_metric in sorted_metric_list:
                if sorted_metric[0] in top_classifier_dict.keys():
                    top_classifier_dict[sorted_metric[0]] = top_classifier_dict[sorted_metric[0]] + 1
                else:
                    top_classifier_dict[sorted_metric[0]] = 1
    return top_classifier_dict


def get_top_classifier_given_category(dataframe,metric_name='accuracy',top_n=1,category_name_list=None,vectorizer=None):
    category_names = dataframeutils.get_unique_values_from_dataframe(dataframe,'category_folder_name')
    if category_name_list is not None:
        category_names = filter(lambda x:x in category_name_list,category_names)
    category_top_classifier_dict = dict({})
    for category_name in category_names:
        filtered_dataframe = dataframe[dataframe['category_folder_name'] == category_name]
        top_classifier_dict = get_top_classifier_dict(filtered_dataframe, metric_name=metric_name,
                                                      top_n=top_n, vectorizer=vectorizer)
        category_top_classifier_dict[category_name] = top_classifier_dict
    return category_top_classifier_dict


def get_top_vectorizer_given_category(dataframe,metric_name='accuracy',top_n=1,category_name_list=None,classifier=None):
    category_names = dataframeutils.get_unique_values_from_dataframe(dataframe, 'category_folder_name')
    if category_name_list is not None:
        category_names = filter(lambda x: x in category_name_list, category_names)
    category_top_vectorizer_dict = dict({})
    for category_name in category_names:
        filtered_dataframe = dataframe[dataframe['category_folder_name'] == category_name]
        top_vectorizer_dict = get_top_vectorizer_dict(filtered_dataframe, metric_name=metric_name,
                                                      top_n=top_n, classifier=classifier)
        category_top_vectorizer_dict[category_name] = top_vectorizer_dict
    return category_top_vectorizer_dict
