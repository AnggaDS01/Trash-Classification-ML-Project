import sys
import inspect
from trashnet.constant.training_pipeline import *
from trashnet.exception import TrashClassificationException
from trashnet.components.data_ingestion import DataIngestion
from trashnet.components.data_transformation import DataTransformation
from trashnet.components.model_trainer import ModelTrainer

from trashnet.entity.config_entity import (DataIngestionConfig,
                                           DataTransformationConfig,
                                           ModelTrainerConfig
                                           )

from trashnet.entity.artifacts_entity import (DataIngestionArtifact,
                                              DataTransformationArtifact)

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return data_ingestion_artifact

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def start_data_transformation(
            self,
            data_ingestion_artifact: DataIngestionArtifact
            )-> DataTransformation:
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            data_transformation = DataTransformation(
                data_ingestion_artifacts = data_ingestion_artifact,
                data_transformation_config = self.data_transformation_config,
            )

            data_transformation_artifact = data_transformation.initiate_data_transformation()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return data_transformation_artifact

        except Exception as e:
            raise TrashClassificationException(e, sys)

    def start_model_trainer(
            self, 
            data_transformation_artifact: DataTransformationArtifact
        ) -> None:
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            model_trainer = ModelTrainer(
                data_transformation_artifact = data_transformation_artifact,
                model_trainer_config = self.model_trainer_config
            )

            model_trainer_artifact = model_trainer.initiate_model_trainer()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return model_trainer_artifact

        except Exception as e:
            raise TrashClassificationException(e, sys)

    def run_pipeline(self) -> None:
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_transformation_artifact = self.start_data_transformation(data_ingestion_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)

            return model_trainer_artifact

        except Exception as e:
            raise TrashClassificationException(e, sys)