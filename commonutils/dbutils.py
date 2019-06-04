from db import mongobase
from constants import dbconstants


def get_mongodb_connection(uri=dbconstants.LOCAL_MONGO_HOSTNAME,port_no=dbconstants.LOCAL_MONGO_PORT,db_name=dbconstants.DB_NAME):
    mongo = mongobase.MongoConnector(uri,port_no)
    mongo.set_db(db_name)
    return mongo


def check_and_insert_in_db(data_dict):
    mongo_connector = get_mongodb_connection()
    mongo_connector.set_collection(dbconstants.COLLECTION_NAME)
    query_dict = dict({'classifier_name': data_dict['classifier_name'],
                       'data_filename': data_dict['data_filename'],
                       'vectorizer': data_dict['vectorizer']})

    if mongo_connector.check_document(query_dict) is False:
        mongo_connector.insert_document(data_dict)
    mongo_connector.close_connection()


def check_for_duplicate(classifier_name,vectorizer_name,data_filename):
    mongo_connector = get_mongodb_connection()
    mongo_connector.set_collection(dbconstants.COLLECTION_NAME)
    query_dict = dict({'classifier_name': classifier_name,
                       'data_filename': data_filename,
                       'vectorizer': vectorizer_name})

    if mongo_connector.check_document(query_dict) is False:
        mongo_connector.close_connection()
        return False
    mongo_connector.close_connection()
    return True


def get_size_filter_query(less_than_value=None,greater_than_value=None,column_name='num_rows'):
    if less_than_value is None and greater_than_value is not None:
        query_dict = dict({column_name:{'$gt': greater_than_value}})
    elif less_than_value is not None and greater_than_value is None:
        query_dict = dict({column_name: {'$lte': less_than_value}})
    elif less_than_value is not None and greater_than_value is not None:
        query_dict = dict({'$and':[{column_name:{'$gt': greater_than_value}},{column_name:{'$lte':less_than_value}}]})
    else:
        query_dict = dict({})
    return query_dict


def get_imbalance_filter_query(lower_range_value,upper_range_value, column_name='imbalance_measure'):
    query_dict = dict({'$and': [{column_name: {'$gte': lower_range_value}}, {column_name: {'$lte': upper_range_value}}]})
    return query_dict
