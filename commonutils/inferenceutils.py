from commonutils import dataframeutils
from constants import commonconstants


def get_summary_statistics(columns_to_keep_list,metric_name_list,less_than_value=None,greater_than_value=None,
                           category_name='all',vectorizer='all',classifier_name='all',metadata_stat=False):
    dataframe = dataframeutils.get_dataframe_size_filter(less_than_value=less_than_value,
                                                         greater_than_value=greater_than_value)
    columns_to_keep_list.extend(metric_name_list)
    for metric_name in metric_name_list:
        dataframe[metric_name] = list(map(lambda x: x*100.0, dataframe[metric_name].values))
    if category_name != 'all':
        dataframe = dataframe[dataframe['category_folder_name'] == category_name]
    if vectorizer != 'all':
        dataframe = dataframe[dataframe['vectorizer'] == vectorizer]
    if classifier_name != 'all':
        dataframe = dataframe[dataframe['classifier_name'] == classifier_name]
    if metadata_stat:
        dataframe.drop_duplicates(subset=['data_filename'], keep='first', inplace=True)
    dataframe = dataframe[columns_to_keep_list]
    return dataframe.describe()


def get_global_statistics():
    for vectorizer in commonconstants.VECTORIZER_LIST:
        print("----------- For Vectorizer %s -----------\n" % vectorizer)
        print(get_summary_statistics(['vectorizer'],commonconstants.METRIC_LIST,
                                     vectorizer=vectorizer))
        print("----------------------------------------------\n")

    for classifier_name in commonconstants.CLASSIFIER_LIST:
        print("------------- For Classifier %s ------------\n" % classifier_name)
        print(get_summary_statistics(['classifier_name'], commonconstants.METRIC_LIST,
                                     classifier_name=classifier_name))
        print("------------------------------------------------\n")


def get_statistics_by_category(less_than_size=None,greater_than_size=None):
    for category_name in commonconstants.CATEGORY_LIST:
        for vectorizer in commonconstants.VECTORIZER_LIST:
            print("------------- For Category %s and Vectorizer %s ----------\n" % (category_name, vectorizer))
            print(get_summary_statistics(['vectorizer'],commonconstants.METRIC_LIST,
                                         category_name=category_name, vectorizer=vectorizer,
                                         less_than_value=less_than_size, greater_than_value=greater_than_size))
            print('-------------------------------------------------\n')

    for category_name in commonconstants.CATEGORY_LIST:
        for classifier_name in commonconstants.CLASSIFIER_LIST:
            print("------------ For Category %s and Classifier %s ----------\n" % (category_name, classifier_name))
            print(get_summary_statistics(['classifier_name'], commonconstants.METRIC_LIST,
                                         category_name=category_name, classifier_name=classifier_name,
                                         less_than_value=less_than_size, greater_than_value=greater_than_size))
            print('----------------------------------------------------\n')


def get_metadata_statistics_by_category():
    for category_name in commonconstants.CATEGORY_LIST:
        print("-------- Metadata for Category %s ---------" % category_name)
        print(get_summary_statistics(['category_folder_name'], commonconstants.METADATA_COLUMN_LIST,
                                     category_name=category_name,metadata_stat=True))


def get_imbalance_measure_statistics_by_category(lower_range_value, higher_range_value):
    dataframe = dataframeutils.get_dataframe_for_imbalance_range(lower_range_value, higher_range_value)
    category_list = commonconstants.CATEGORY_LIST
    metric_list = commonconstants.METRIC_LIST
    classifier_list = commonconstants.CLASSIFIER_LIST
    vectorizer_list = commonconstants.VECTORIZER_LIST

    for metric_name in metric_list:
        dataframe[metric_name] = list(map(lambda x: x*100.0, dataframe[metric_name].values))

    for classifier in classifier_list:
        filtered_dataframe = dataframe[dataframe['classifier_name'] == classifier]
        filtered_dataframe = filtered_dataframe[metric_list]
        print('--------- For Classifier %s Imbalance Measure %s to %s' %(classifier,str(lower_range_value),str(higher_range_value)))
        print(filtered_dataframe.describe())
        print('----------------------------\n')

    for category in category_list:
        for vectorizer in vectorizer_list:
            filtered_dataframe = dataframe[(dataframe['category_folder_name'] == category) & (dataframe['vectorizer'] == vectorizer)]
            filtered_dataframe = filtered_dataframe[metric_list]
            print('------------- For Category %s ---------' % category)
            print('--------- For Vectorizer %s Imbalance Measure %s to %s' % (vectorizer, str(lower_range_value), str(higher_range_value)))
            print(filtered_dataframe.describe())
            print('----------------------------\n')


def get_data_size_stats_category(less_than_value=None, greater_than_value=None):
    dataframe = dataframeutils.get_dataframe_size_filter(less_than_value=less_than_value,
                                                         greater_than_value=greater_than_value)
    dataframe.drop_duplicates(subset=['data_filename'], keep='first', inplace=True)
    for category_name in commonconstants.CATEGORY_LIST:
        filtered_frame = dataframe[dataframe['category_folder_name'] == category_name]
        print('------- For Category Name : %s --------------\n' % category_name)
        print(filtered_frame['num_class_labels'].value_counts())
        print('---------------------------------\n')


def get_data_size_stats_imbalance_category(lower_range=None,upper_range=None):
    dataframe = dataframeutils.get_dataframe_for_imbalance_range(lower_range_value=lower_range,
                                                                 upper_range_value=upper_range)
    dataframe.drop_duplicates(subset=['data_filename'], keep='first', inplace=True)
    print('********* For Imbalance Measure Range %s to %s *********' % (str(lower_range), str(upper_range)))
    print(len(dataframe))
    print(dataframe['category_folder_name'].value_counts())
    print('---------------------------------\n')


get_statistics_by_category(greater_than_size=50000)
