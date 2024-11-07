import sys
import inspect
from trashnet.exception import TrashClassificationException
from trashnet.components.data_ingestion import DataIngestion
from trashnet.components.data_transformation import DataTransformation
from trashnet.components.model_trainer import ModelTrainer
from trashnet.components.model_evaluation import ModelEvaluation

from trashnet.configuration.configuration import ConfigurationManager


from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text)

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_transformation_config = config.get_data_transformation_config()
        self.model_trainer_config = config.get_model_trainer_config()
        self.model_evaluation_config = config.get_model_evaluation_config()
        self.wandb_config = config.get_wandb_config()

    
    def start_data_ingestion(self):
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion_run = data_ingestion.initiate_data_ingestion()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return data_ingestion_run

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def start_data_transformation(
            self,
            data_ingestion_config
        ):
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            data_transformation = DataTransformation(
                data_ingestion_config = data_ingestion_config,
                data_transformation_config = self.data_transformation_config,
            )

            data_transformation_run = data_transformation.initiate_data_transformation()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return data_transformation_run

        except Exception as e:
            raise TrashClassificationException(e, sys)

    def start_model_trainer(
            self, 
            data_transformation_artifact
        ) :
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            model_trainer = ModelTrainer(
                data_transformation_config = data_transformation_artifact,
                model_trainer_config = self.model_trainer_config,
                wandb_config = self.wandb_config
            )

            model_trainer_config = model_trainer.initiate_model_trainer()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

    def start_model_evaluation(
            self,
            data_transformation_config, 
            model_trainer_config
        ):
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            model_trainer = ModelEvaluation(
                data_transformation_config = data_transformation_config,
                model_trainer_config = model_trainer_config,
                model_evaluation_config = self.model_evaluation_config,
                # wandb_config = self.wandb_config
            )

            model_trainer_config = model_trainer.initiate_model_trainer()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
        

    def run_pipeline(self) -> None:
        try:
            data_ingestion_config = self.start_data_ingestion()
            data_transformation_config = self.start_data_transformation(data_ingestion_config)
            model_trainer_config = self.start_model_trainer(data_transformation_config)
            model_evaluation_config = self.start_model_evaluation(data_transformation_config, model_trainer_config)

            return model_evaluation_config

        except Exception as e:
            raise TrashClassificationException(e, sys)