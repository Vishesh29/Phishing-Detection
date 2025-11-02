from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging
from network_security.components.data_ingestion import DataIngestion
from network_security.components.data_validation import DataValidation
from network_security.components.data_transformation import DataTransformation
from network_security.components.data_trainer_model import ModelTrainer
from network_security.entity.config_entity import DataIngestionConfig, TrainingPipelineConfig, DataValidationConfig, DataTransformationConfig, DataModelTrainerConfig
import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        data_validation_config = DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation_config = DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_model_trainer_config = DataModelTrainerConfig(training_pipeline_config=training_pipeline_config)

        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        logging.info("Starting data ingestion process")
        data_ingestion_artifact = data_ingestion.initiaize_data_ingestion()
        logging.info("Data ingestion process completed")

        logging.info("Starting data validation process")
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        data_validation_artifact = data_validation.initialize_data_validation()
        logging.info("Data validation process completed")

        logging.info("Starting Data Transformation process")
        data_transformation = DataTransformation(data_validation_artifact, data_transformation_config)
        data_transformation_artifact = data_transformation.initialize_data_transformation()
        logging.info("Data Transformation process completed")

        logging.info("Starting Model Trainer process")
        data_model_trainer = ModelTrainer(data_transformation_artifact=data_transformation_artifact,
                                          model_trainer_config=data_model_trainer_config)
        model_trainer_artifact = data_model_trainer.initalize_model_trainer()
        logging.info("Model Trainer process completed")

        
    except Exception as e:
        logging.error(f"Error occurred during data ingestion: {e}")
        raise NetworkSecurityException(e, sys)

