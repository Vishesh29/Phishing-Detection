## Data validation process -> 
### 1. Same schema means same no of features.
### 2. Data Drift. 3. Validate number of features.

from network_security.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from network_security.entity.config_entity import DataValidationConfig
from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.train_pipeline import SCHEMA_FILE_PATH
from network_security.utils.util import read_yaml_file, write_yaml_file

import numpy as np
import pandas as pd
import os, sys
from scipy.stats import ks_2samp # two sample test for goodness of fit


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_file_path = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def validate_number_of_features(self,df: pd.DataFrame) -> bool:
        try:
            num_of_features = len(self._schema_file_path)
            logging.info(f"Required number of features: {num_of_features}")
            logging.info(f"Dataframe has columns: {df.columns}")
            if num_of_features == len(df.columns):
                return True
            return False
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def numerical_columns(self, df: pd.DataFrame) -> list:
        try:
            numerical_cols = []
            numerical_cols_schema = self._schema_file_path.get("numerical_columns", [])
            for feature in numerical_cols_schema:
                if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                    numerical_cols.append(feature)
            return numerical_cols
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    
    def detect_dataset_drift(self,base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}
            for column in base_df.columns:
                df1 = base_df[column]
                df2 = current_df[column]
                is_same_sample_dist = ks_2samp(df1, df2)
                if threshold <= is_same_sample_dist.pvalue:
                    is_found = False # null hypothesis accepted
                else:
                    is_found = True
                    status = False
                report.update({column: {"pvalue": float(is_same_sample_dist.pvalue), "drift_status": is_found}})
        
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)
            write_yaml_file(drift_report_file_path,report)

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


    def initialize_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            ## read the data
            train_df = self.read_data(train_file_path)
            test_df = self.read_data(test_file_path)

            ## validate number of features
            train_status =  self.validate_number_of_features(train_df)
            test_status =  self.validate_number_of_features(test_df)

            if not (train_status or test_status):
                error_msg = f"Dataframe does not contain all columns. \n"
            
            ## check numerical columns
            numerical_train_cols = self.numerical_columns(train_df)
            logging.info(f"Numerical columns: {numerical_train_cols}")
            if len(numerical_train_cols) == 0:
                raise Exception(f"No numerical columns found in the data")
            
            numerical_test_cols = self.numerical_columns(test_df)
            logging.info(f"Numerical columns: {numerical_test_cols}")
            if len(numerical_test_cols) == 0:
                raise Exception(f"No numerical columns found in the data")


            ## detect data drift
            status = self.detect_dataset_drift(base_df = train_df, current_df = test_df)
            if not status:
                logging.error("Data drift found")
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path

            )
            logging.info(f"Data Validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e


