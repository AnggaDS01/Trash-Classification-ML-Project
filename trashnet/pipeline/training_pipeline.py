import sys
import inspect
from trashnet.exception import TrashClassificationException
from trashnet.configuration.configuration import ConfigurationManager

from trashnet.components.data_ingestion_components.data_ingestion import DataIngestion
from trashnet.components.data_preprocessing_components.data_preprocessing import DataPreprocessing
from trashnet.components.model_trainer_components.model_trainer import ModelTrainer
from trashnet.components.model_evaluation_components.model_evaluation import ModelEvaluation
from trashnet.components.model_pusher_components.model_pusher import ModelPusher

from trashnet.entity.config_entity import (DataIngestionConfig, 
                                           DataPreprocessingConfig,
                                           ModelTrainerConfig,
                                           ModelEvaluationConfig)

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text)

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        self.model_trainer_config = config.get_model_trainer_config()
        self.model_evaluation_config = config.get_model_evaluation_config()
        self.model_pusher_config = config.get_model_pusher_config()
        self.wandb_config = config.get_wandb_config()

    
    def start_data_ingestion(self) -> DataIngestionConfig:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
            )

            data_ingestion.initiate_data_ingestion()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

            return self.data_ingestion_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def start_data_preprocessing(
            self,
            data_ingestion_config: DataIngestionConfig
        ) -> DataPreprocessingConfig:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            data_preprocessing = DataPreprocessing(
                data_ingestion_config = data_ingestion_config,
                data_preprocessing_config = self.data_preprocessing_config,
            )

            data_preprocessing.initiate_data_preprocessing()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

            return self.data_preprocessing_config

        except Exception as e:
            raise TrashClassificationException(e, sys)

    def start_model_trainer(
            self, 
            data_preprocessing_config: DataPreprocessingConfig
        ) -> ModelTrainerConfig:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            model_trainer = ModelTrainer(
                data_preprocessing_config = data_preprocessing_config,
                model_trainer_config = self.model_trainer_config,
                wandb_config = self.wandb_config
            )

            model_trainer.initiate_model_trainer()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

            return self.model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

    def start_model_evaluation(
            self,
            data_preprocessing_config: DataPreprocessingConfig, 
            model_trainer_config: ModelTrainerConfig
        ) -> ModelEvaluationConfig:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            model_evaluation = ModelEvaluation(
                data_preprocessing_config = data_preprocessing_config,
                model_trainer_config = model_trainer_config,
                model_evaluation_config = self.model_evaluation_config,
                # wandb_config = self.wandb_config
            )

            model_evaluation.initiate_model_evaluation()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

            return self.model_evaluation_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

    def start_model_pusher(
            self,
            model_trainer_config: ModelTrainerConfig,
        ) -> None:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            model_pusher = ModelPusher(
                model_trainer_config = model_trainer_config,
                model_pusher_config = self.model_pusher_config,
            )

            model_pusher.initiate_model_pusher()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

    def run_pipeline(self, stage=None) -> None:
        try:
            if stage == "data_ingestion":
                self.start_data_ingestion()
                
            elif stage == "data_preprocessing":
                self.start_data_preprocessing(self.data_ingestion_config)

            elif stage == "model_training_and_evaluation":
                self.start_model_trainer(self.data_preprocessing_config)
                self.start_model_evaluation(self.data_preprocessing_config, self.model_trainer_config)

            elif stage == "model_pusher":
                self.start_model_pusher(self.model_trainer_config)

            elif stage is None:
                self.start_data_ingestion()
                self.start_data_preprocessing(self.data_ingestion_config)
                self.start_model_trainer(self.data_preprocessing_config)
                self.start_model_evaluation(self.data_preprocessing_config, self.model_trainer_config)
                self.start_model_pusher(self.model_trainer_config)
                display_log_message("Pipeline selesai.")
            else:
                print(f"Unknown stage: {stage}")

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else None
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline(stage)