from datetime import datetime
import os
from network_security.constants import train_pipeline


class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%Y-%m-%d-%H-%M-%S")
        self.pipeline_name = train_pipeline.PIPELINE_NAME
        self.artifact_name = train_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, timestamp)
        self.model_dir=os.path.join("final_models")
        self.timestamp = timestamp
        

class DataIngestionConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, train_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, train_pipeline.FILE_NAME)
        self.train_file_path: str = os.path.join(self.data_ingestion_dir, train_pipeline.TRAIN_FILE_NAME)
        self.test_file_path: str = os.path.join(self.data_ingestion_dir, train_pipeline.TEST_FILE_NAME)
        self.train_test_split_ratio: float = train_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = train_pipeline.DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = train_pipeline.DATA_INGESTION_DATABASE_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, train_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir : str = os.path.join(self.data_validation_dir, train_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, train_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path: str = os.path.join(self.valid_data_dir, train_pipeline.TRAIN_FILE_NAME)
        self.valid_test_file_path: str = os.path.join(self.valid_data_dir, train_pipeline.TEST_FILE_NAME)
        self.invalid_train_file_path: str = os.path.join(self.invalid_data_dir, train_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path: str = os.path.join(self.invalid_data_dir, train_pipeline.TEST_FILE_NAME)
        self.drift_report_file_path: str = os.path.join(self.data_validation_dir, train_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR, 
                                                        train_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)


class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, train_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_train_file_path: str = os.path.join(self.data_transformation_dir, train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                              train_pipeline.TRAIN_FILE_NAME.replace("csv","npy"))
        self.transformed_test_file_path: str = os.path.join(self.data_transformation_dir, train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                              train_pipeline.TEST_FILE_NAME.replace("csv","npy"))
        self.transformed_obj_file_path: str = os.path.join(self.data_transformation_dir, train_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJ_DIR, 
                                                            train_pipeline.PREPROCESSING_OBJ_FILE_NAME)
        

class DataModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, train_pipeline.MODEL_TRAINER_DIR)
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir, train_pipeline.MODEL_TRAINER_DIR, train_pipeline.MODEL_TRAINER_FILE_NAME)
        self.expected_accuracy: float = train_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_threshold: float = train_pipeline.MODEL_TRAINER_OVERFITTING_THRESHOLD
        self.final_model_file_path: str = os.path.join(train_pipeline.FINAL_MODEL_DIR, train_pipeline.FINAL_MODEL_FILE_NAME)
    