from commonutils import dataframeutils, visualizationutils
from constants import commonconstants, modelconstants
import os


def get_filename_to_save(names_to_append, base_location=commonconstants.PAPER_FIGURES_PATH,default_extension='.png'):
    filename = ''
    for name in names_to_append:
        filename = filename + name + '_'
    filename = filename[:-1] + default_extension
    return os.path.join(base_location, filename)


def get_global_metric_visualization(less_than_value=None, greater_than_value=None):
    dataframe = dataframeutils.get_dataframe_size_filter(less_than_value=less_than_value,
                                                         greater_than_value=greater_than_value)
    dataframe['classifier_name'] = list(map(lambda x: modelconstants.CLASSIFIER_ALIAS_DICT[x],
                                            dataframe['classifier_name'].values))
    metrics_list = ['accuracy','macro_precision','macro_recall','macro_f1']
    plotting_object = ['vectorizer','classifier_name']
    for plot_object in plotting_object:
        for metric_name in metrics_list:
            box_plot_filename = get_filename_to_save(['boxplot',plot_object,metric_name,
                                                      str(less_than_value),str(greater_than_value)])
            violin_plot_filename = get_filename_to_save(['violin',plot_object,metric_name,
                                                         str(less_than_value),str(greater_than_value)])
            visualizationutils.plot_boxplot_chart(x_name=plot_object,y_name=metric_name,
                                                  dataframe=dataframe,filename_to_save=box_plot_filename)
            visualizationutils.plot_violinstrip_chart(x_name=plot_object, y_name=metric_name,
                                                      dataframe=dataframe, filename_to_save=violin_plot_filename)


def get_global_imbalance_metric_visualization(lower_range_value=None, upper_range_value=None):
    dataframe = dataframeutils.get_dataframe_for_imbalance_range(lower_range_value=lower_range_value,
                                                     upper_range_value=upper_range_value)
    dataframe['classifier_name'] = list(map(lambda x: modelconstants.CLASSIFIER_ALIAS_DICT[x],
                                            dataframe['classifier_name'].values))
    metrics_list = ['accuracy','macro_precision','macro_recall','macro_f1']
    plotting_object = ['vectorizer','classifier_name']
    for plot_object in plotting_object:
        for metric_name in metrics_list:
            box_plot_filename = get_filename_to_save(['boxplot',plot_object,metric_name,
                                                      str(lower_range_value),str(upper_range_value)])
            violin_plot_filename = get_filename_to_save(['violin',plot_object,metric_name,
                                                         str(lower_range_value),str(upper_range_value)])
            visualizationutils.plot_boxplot_chart(x_name=plot_object,y_name=metric_name,
                                                  dataframe=dataframe,filename_to_save=box_plot_filename)
            visualizationutils.plot_violinstrip_chart(x_name=plot_object, y_name=metric_name,
                                                      dataframe=dataframe, filename_to_save=violin_plot_filename)


def get_metric_visualization_by_category(less_than_value=None,greater_than_value=None,category_name='all'):
    dataframe = dataframeutils.get_dataframe_size_filter(less_than_value=less_than_value,
                                                         greater_than_value=greater_than_value)
    dataframe['classifier_name'] = list(map(lambda x: modelconstants.CLASSIFIER_ALIAS_DICT[x],
                                            dataframe['classifier_name'].values))
    category_list = commonconstants.CATEGORY_LIST
    if category_name != 'all':
        category_list = filter(lambda x: x in category_list,category_name)
    for category in category_list:
        filtered_dataframe = dataframe[dataframe['category_folder_name'] == category]
        metrics_list = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
        plotting_object = ['vectorizer', 'classifier_name']
        for plot_object in plotting_object:
            for metric_name in metrics_list:
                box_plot_filename = get_filename_to_save(['boxplot', plot_object, metric_name,category,
                                                          str(less_than_value), str(greater_than_value)])
                violin_plot_filename = get_filename_to_save(['violin', plot_object, metric_name,category,
                                                             str(less_than_value), str(greater_than_value)])
                visualizationutils.plot_boxplot_chart(x_name=plot_object, y_name=metric_name,
                                                      dataframe=filtered_dataframe, filename_to_save=box_plot_filename)
                visualizationutils.plot_violinstrip_chart(x_name=plot_object, y_name=metric_name,
                                                          dataframe=filtered_dataframe, filename_to_save=violin_plot_filename)


def get_global_metadata_visualization(less_than_value=None,greater_than_value=None):
    dataframe = dataframeutils.get_dataframe_size_filter(less_than_value=less_than_value,
                                                         greater_than_value=greater_than_value)
    unique_num_rows = dataframeutils.get_unique_values_from_dataframe(dataframe, column_name='num_rows')
    unique_average_sentence_length = dataframeutils.get_unique_values_from_dataframe(dataframe,
                                                                                     column_name='average_sentence_length')
    unique_class_imbalance = dataframeutils.get_unique_values_from_dataframe(dataframe,
                                                                             column_name='imbalance_measure')
    unique_class_labels = list(dataframe['num_class_labels'].values)
    unique_num_tokens = dataframeutils.get_unique_values_from_dataframe(dataframe,
                                                                        column_name='num_tokens')
    histogram_plot_iterable = [unique_num_rows, unique_average_sentence_length, unique_class_imbalance,
                                unique_class_labels, unique_num_tokens]
    title_iterable = ['Distribution of rows','Distribution of Average Sentence Length',
                     'Distribution of Class Imbalance','Distribution of Class Labels','Distribution of Tokens']
    x_label_iterable = ['Number of Rows','Average Sentence Length','Class Imbalance','Number of Class Labels',
                        'Number of Tokens']
    y_label_iterable = ['Count'] * len(x_label_iterable)

    for plot_iter, title_iter, x_label_iter, y_label_iter in zip(histogram_plot_iterable, title_iterable,
                                                                 x_label_iterable, y_label_iterable):

        filename_to_save = get_filename_to_save([str(title_iter)])
        visualizationutils.plot_histogram_chart(plot_iter, title_name=title_iter,
                                                x_label=x_label_iter, y_label=y_label_iter,
                                                filename_to_save=filename_to_save)
