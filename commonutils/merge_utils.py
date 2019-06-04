import pandas as pd


def merge_dicts(dict_one, dict_two):
    merged_dict = {**dict_one, **dict_two}
    return merged_dict


def get_complete_dataframe_from_dict(dict_list):
    dataframe_list = list([])
    for dictionary_object in dict_list:
        dataframe_list.append(pd.DataFrame(dictionary_object))
    return pd.concat(dataframe_list, axis=0, ignore_index=False)


def convert_word_weight_tuple_list_to_dict(word_weight_tuple_list,vectorizer_name):
    word_list = list([])
    weight_list = list([])
    vectorizer_name_list = list([])
    for word_weight_tuple in word_weight_tuple_list:
        word_list.append(word_weight_tuple[0])
        weight_list.append(word_weight_tuple[1])
        vectorizer_name_list.append(vectorizer_name)
    return dict({'words': word_list, 'weights': weight_list, 'vectorizer': vectorizer_name_list})
