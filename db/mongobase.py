# -*- coding: utf-8 -*-
"""
Created on Thu Dec 05 16:20:10 2018
Defines the base class for all interactions with Mongo DB
"""

import pymongo
from constants import dbconstants


class MongoConnector:
    
    """
    Initializes the mongo connector with database uri and port number
    """
    def __init__(self,uri,port_no):
        self.uri = uri
        self.port_no = port_no
        self.mongo_client = pymongo.MongoClient(uri,port_no)
        self.collection = None
        self.db = None
    
    """
    Sets the database for a particular mongoclient 
    
    Params:
    -------
    db_name - Name of the database to be connected to
    
    Returns:
    --------
    None
    """
    def set_db(self,db_name):
        self.db = self.mongo_client.get_database(db_name)

    """
    Returns the database for a particular mongoclient 
    
    Returns:
    --------
    database object containing the particulars
    """
    def get_db(self):
        return self.db 
    
    """
    Sets the collection for a given mongoclient
    
    Params:
    -------
    collection_name - String containing the length of the connection which is to be used
    """
    def set_collection(self,collection_name):
         self.collection = self.db[collection_name]  

    def create_index(self,index_field_name='inserted_timestamp',expire_time_seconds=dbconstants.EXPIRE_TIME):
        self.collection.create_index(index_field_name,expireAfterSeconds=expire_time_seconds)

    """
    Returns the collection for a given mongoclient
    
    Params:
    -------
    collection_name - String containing the length of the connection which is to be used
    """
    def get_collection(self):
        return self.collection
    
    """
    Inserts a given document in the collection
    
    Params:
    --------
    document - the document object which is to be inserted in the database
    """
    def insert_document(self,document):
        self.collection.insert(document)
    
    """
    Updates a given document using the specification
    
    Params:
    -------
    update_spec - Object containing the update specifications
    document - document query matching the updating document
    """
    def update_document(self,update_spec,document):
        self.collection.update(document, update_spec)
    
    """
    Finds a document in the database matching a given query
    
    Params:
    --------
    query - dictionary containing the query for the document
    
    Returns:
    --------
    cursor pointing to the matching set of documents
    """
    def find_document(self,query):
        return self.collection.find(query)
    
    """
    Deletes all the documents matching a given query
    
    Params:
    --------
    query - Dictionary containing the query to be executed
    
    Returns:
    --------
    Nothing just deletes the matching documents
    """
    def delete_document(self,query):
        self.collection.delete_many(query) 
    
    """
    Checks if a given document already exists in the database or not
    
    Params:
    -------
    document - Query which is to be matched
    """
    def check_document(self,document):
        cursor = self.collection.find(document, projection={'_id':True})
        if cursor.count() > 0:
            cursor.close()
            return True
        cursor.close()
        return False

    def close_connection(self):
        self.mongo_client.close()