import os
import sys
import mlflow

from network_security.exception.exception import NetworkSecurityException
from network_security.logging.logger import logging

from network_security.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from network_security.entity.config_entity import DataModelTrainerConfig

from network_security.utils.util import load_numpy_array_data, save_object, load_object, save_numpy_array_data, evaluate_model
from network_security.utils.classification_metrics import get_classification_score
from network_security.utils.model_estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier)

## use Dagshub for version control of models and datasets in remote repository
## use mlflow for version control of models in local repository

#import dagshub
#dagshub.init(repo_owner='<username>', repo_name='<repo_name>', mlflow=True)




class ModelTrainer:
    def __init__(self, model_trainer_config: DataModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, train_classification_metric):
        try:
            with mlflow.start_run():
                f1_score = train_classification_metric.f1_score
                accuracy = train_classification_metric.accuracy_score
                precision = train_classification_metric.precision_score
                recall = train_classification_metric.recall_score

                mlflow.log_metric("F1_Score", f1_score)
                mlflow.log_metric("Accuracy", accuracy)
                mlflow.log_metric("Precision", precision)
                mlflow.log_metric("Recall", recall)
            
                mlflow.sklearn.log_model(model, "model")                   

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "LogisticRegression": LogisticRegression(verbose=1),
                "DecisionTree": DecisionTreeClassifier(),
                "RandomForest": RandomForestClassifier(verbose=1),
                "GradientBoosting": GradientBoostingClassifier(verbose=1),
                "AdaBoost": AdaBoostClassifier()
            }

            params = {
                "LogisticRegression": {
                    'C': [0.1, 1.0, 10, 100],
                    'solver': ['liblinear', 'saga']
                },
                "DecisionTree": {
                    'criterion': ['gini', 'entropy', 'log_loss'],
                    #'max_depth': [None, 10, 20, 30],
                    #'min_samples_split': [2, 5, 10]
                },
                "RandomForest": {
                    'n_estimators': [10,20,50, 100, 200],
                    #'criterion': ['gini', 'entropy'],
                    #'max_depth': [None, 10, 20],
                    #'min_samples_split': [2, 5, 10]
                },
                "GradientBoosting": {
                    'n_estimators': [10,50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.4, 0.6, 0.8, 1.0],
                    #'max_depth': [3, 5, 7]
                },
                "AdaBoost": {
                    'n_estimators': [10, 50, 100, 200],
                    'learning_rate': [0.01, 0.1, 1.0]
                }
            }

            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)

            # Get best model score from dict
            best_model_info = max(model_report.values(), key=lambda x: x['test_metric'])
            best_model_score = best_model_info['test_metric']
            ## Get best model name from dict
            best_model_name = [k for k, v in model_report.items() if v is best_model_info][0]

            best_model = models[best_model_name]
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_classification_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
            test_classification_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            ## track experiments with the MLFlow -> Tool to manage entire Data Science project life cycle
            self.track_mlflow(best_model, train_classification_metric)
            self.track_mlflow(best_model, test_classification_metric)

            ## ----------------------------------

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_obj_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)

            network_model= NetworkModel(preprocessor=preprocessor, model= best_model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=network_model)

            save_object(self.model_trainer_config.final_model_file_path, obj=best_model)

            ## Model trainer artifact

            model_trainer_artifact = ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                                                          train_metric_artifact=train_classification_metric,
                                                          test_metric_artifact=test_classification_metric)
            
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    
    def initalize_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            ## loading the train and test array
            train_array = load_numpy_array_data(train_file_path)
            test_array = load_numpy_array_data(test_file_path)

            X_train, y_train = train_array[:,:-1], train_array[:,-1]
            X_test, y_test = test_array[:,:-1], test_array[:,-1]

            model_trainer_artifact = self.train_model(X_train, y_train,X_test, y_test)
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)



