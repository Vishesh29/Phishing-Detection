import sys,os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.constants.train_pipeline import TARGET_COLUMN, DATA_TRANSFORMATION_IMPUTER_PARAMS
from network_security.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
from network_security.entity.config_entity import DataTransformationConfig
from network_security.utils.util import save_numpy_array_data, save_object

class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e,sys)

    def get_data_transformer_object(self) -> Pipeline:
        """It initializes the KNNImputer object and returns the pipeline object"""
        try:
            logging.info("Initializing the KNNImputer object")
            knn_imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor = Pipeline(steps=[("imputer",knn_imputer)])
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys)



    def initialize_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            logging.info("Reading validated train and test data")
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            
            logging.info("Splitting input and target feature from both train and test data")
            ## training data
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            ## testing data
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]

            ## Replace -1 with 0
            input_feature_train_df.replace(to_replace=-1, value=0, inplace=True)
            input_feature_test_df.replace(to_replace=-1, value=0, inplace=True)

            ## KNN Imputer will replace the missing values with avg of nearest neighbour values
            preprocessor = self.get_data_transformer_object()
            preprocessor_obj = preprocessor.fit(input_feature_train_df)
            transformed_input_train_feature = preprocessor_obj.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)] ## combine input and target feature
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            ## save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            save_object(self.data_transformation_config.transformed_obj_file_path, obj= preprocessor_obj)

            save_object("final_models/preprocessor.pkl", obj= preprocessor_obj)

            ## prepare artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_obj_file_path=self.data_transformation_config.transformed_obj_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        