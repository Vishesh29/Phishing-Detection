import os
import sys
import json
import certifi
import pandas as pd
import numpy as np
import pymongo
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from dotenv import load_dotenv
load_dotenv()

MONDO_DB_URL = os.getenv("MONGO_DB_URL")
ca = certifi.where() # Get the path to the CA ( Certificate Authorities) bundle for SSL verification


class NetworkDataExtract():
    def __init__(self, records, database, collection):
        try:
            self.records = records
            self.database = database
            self.collection = collection
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def convert_csv_to_json(self, csv_file_path):
        try:
            df = pd.read_csv(csv_file_path)
            df.reset_index(drop=True, inplace=True)
            records =  list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def insert_data_to_mongodb(self, records, database, collection): ## collection is table
        try:
            self.mongo_client = pymongo.MongoClient(MONDO_DB_URL, tlsCAFile=ca)
            database = self.mongo_client[database]
            collection = database[collection]
            if isinstance(records, list):
                collection.insert_many(records)
            else:
                collection.insert_one(records)
            logging.info("Data inserted successfully into MongoDB")
            return len(records)
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

if __name__ == "__main__":
    FILE_PATH = "network_data/phisingData.csv"
    DATABASE = "Vishesh"
    COLLECTION = "NetworkData"
    network_data = NetworkDataExtract(records=None, database=DATABASE, collection=COLLECTION)

    records = network_data.convert_csv_to_json(csv_file_path=FILE_PATH)
    no_of_records = network_data.insert_data_to_mongodb(records=records, database=DATABASE, collection=COLLECTION)

    print(f"Total number of records inserted: {no_of_records}")
    