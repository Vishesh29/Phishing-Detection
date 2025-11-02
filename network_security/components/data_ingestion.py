from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

## configuration for data ingestion config
from network_security.entity.config_entity import DataIngestionConfig
from network_security.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import Optional, List
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv
load_dotenv()

MONGODB_URL = os.getenv("MONGO_DB_URL")

print("MONGO_DB_URL:", os.getenv("MONGO_DB_URL"))

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    def export_collection_as_dataframe(self):
        """
        Read data from mongodb
        """
        try:
            db_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGODB_URL)
            collection = self.mongo_client[db_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], inplace=True)
            df.replace({"na": np.nan}, inplace=True)
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)


    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        """
        Save the dataframe into feature store file
        """
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            ## create directory if not exists
            dir_path = os.path.dirname(feature_store_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_path, index=False, header=True)
            logging.info(f"Data exported to feature store at {feature_store_path}")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        """
        Split the data into train and test
        """
        try:
            train_set, test_set = train_test_split(dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42)

            logging.info("Splitting data into train and test sets")
            dir_path = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info('Exporting train and test data')
            train_set.to_csv(self.data_ingestion_config.train_file_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_file_path, index=False, header=True)

        except Exception as e:
            raise NetworkSecurityException(e, sys)



    def initiaize_data_ingestion(self):
        try:
            dataframe = self.export_collection_as_dataframe() 
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            logging.info("Data ingestion completed successfully")

            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.train_file_path,
                                                            test_file_path=self.data_ingestion_config.test_file_path)
            
            return data_ingestion_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        

    
