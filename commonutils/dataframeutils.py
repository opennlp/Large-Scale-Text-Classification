import pandas as pd
from random import randint
from commonutils import dbutils
from constants import  dbconstants


def get_data_frame(filepath):
    df = pd.read_csv(filepath, error_bad_lines=False, encoding='ISO-8859-1')
    df.dropna(inplace=True)
    return df


def get_random_integer_list(start, stop, length):
    random_set = set()
    while len(random_set) < length:
        random_set.add(randint(start, stop))
    random_num_list = list(random_set)
    return random_num_list


def get_dataframe_size_filter(less_than_value=None,greater_than_value=None,column_name='num_rows'):
    query_dict = dbutils.get_size_filter_query(less_than_value,greater_than_value,column_name)
    document_list = list([])
    mongo_connector = dbutils.get_mongodb_connection()
    mongo_connector.set_collection(dbconstants.COLLECTION_NAME)
    document_cursor = mongo_connector.find_document(query_dict)
    for document in document_cursor:
        document_list.append(document)
    return pd.DataFrame(document_list)


def get_unique_values_from_dataframe(dataframe,column_name):
    return list(set(dataframe[column_name].values))


def get_dataframe_for_imbalance_range(lower_range_value, upper_range_value):
    query_dict = dbutils.get_imbalance_filter_query(lower_range_value, upper_range_value)
    document_list = list([])
    mongo_connector = dbutils.get_mongodb_connection()
    mongo_connector.set_collection(dbconstants.COLLECTION_NAME)
    document_cursor = mongo_connector.find_document(query_dict)
    for document in document_cursor:
        document_list.append(document)
    return pd.DataFrame(document_list)
