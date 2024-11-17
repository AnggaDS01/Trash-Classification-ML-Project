import sys
import inspect
from trashnet.exception import TrashClassificationException
from trashnet.configuration.configuration import ConfigurationManager

from trashnet.components.data_ingestion_components.data_ingestion import DataIngestion
from trashnet.components.data_preprocessing_components.data_preprocessing import DataPreprocessing
# from trashnet.components.model_trainer import ModelTrainer
# from trashnet.components.model_evaluation import ModelEvaluation
# from trashnet.components.model_pusher import ModelPusher

from trashnet.entity.config_entity import (DataIngestionConfig, 
                                           DataPreprocessingConfig)

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       custom_title_print)

class TrainPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        # self.model_trainer_config = config.get_model_trainer_config()
        # self.model_evaluation_config = config.get_model_evaluation_config()
        # self.model_pusher_config = config.get_model_pusher_config()
        # self.wandb_config = config.get_wandb_config()

    
    def start_data_ingestion(self) -> None:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            custom_title_print(f'{class_name}')
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            data_ingestion = DataIngestion(
                data_ingestion_config =  self.data_ingestion_config
            )

            data_ingestion.initiate_data_ingestion()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}\n\n")

            return self.data_ingestion_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def start_data_preprocessing(
            self,
            data_ingestion_config: DataIngestionConfig
        ) -> None:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            custom_title_print(f'{class_name}')
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            data_preprocessing = DataPreprocessing(
                data_ingestion_config = data_ingestion_config,
                data_preprocessing_config = self.data_preprocessing_config,
            )

            data_preprocessing.initiate_data_preprocessing()

            display_log_message(f"Finished {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            return self.data_preprocessing_config

        except Exception as e:
            raise TrashClassificationException(e, sys)

    # def start_model_trainer(
    #         self, 
    #         data_preprocessing_artifact
    #     ) :
    #     try:
    #         function_name, file_name_function = display_function_name(inspect.currentframe())
    #         display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

    #         model_trainer = ModelTrainer(
    #             data_preprocessing_config = data_preprocessing_artifact,
    #             model_trainer_config = self.model_trainer_config,
    #             wandb_config = self.wandb_config
    #         )

    #         model_trainer_config = model_trainer.initiate_model_trainer()

    #         display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

    #         return model_trainer_config

    #     except Exception as e:
    #         raise TrashClassificationException(e, sys)
        

    # def start_model_evaluation(
    #         self,
    #         data_preprocessing_config, 
    #         model_trainer_config
    #     ):
    #     try:
    #         function_name, file_name_function = display_function_name(inspect.currentframe())
    #         display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

    #         model_trainer = ModelEvaluation(
    #             data_preprocessing_config = data_preprocessing_config,
    #             model_trainer_config = model_trainer_config,
    #             model_evaluation_config = self.model_evaluation_config,
    #             # wandb_config = self.wandb_config
    #         )

    #         model_trainer_config = model_trainer.initiate_model_evaluation()

    #         display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

    #         return model_trainer_config

    #     except Exception as e:
    #         raise TrashClassificationException(e, sys)
        

    # def start_model_pusher(
    #         self,
    #         model_trainer_config,
    #     ):
    #     try:
    #         function_name, file_name_function = display_function_name(inspect.currentframe())
    #         display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

    #         model_pusher = ModelPusher(
    #             model_trainer_config = model_trainer_config,
    #             model_pusher_config = self.model_pusher_config,
    #         )

    #         model_pusher.initiate_model_pusher()

    #         display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

    #         return model_trainer_config

    #     except Exception as e:
    #         raise TrashClassificationException(e, sys)
        
        

    def run_pipeline(self) -> None:
        try:
            data_ingestion_config = self.start_data_ingestion()
            data_preprocessing_config = self.start_data_preprocessing(data_ingestion_config)
            # model_trainer_config = self.start_model_trainer(data_preprocessing_config)
            # self.start_model_evaluation(data_preprocessing_config, model_trainer_config)
            # self.start_model_pusher(model_trainer_config)

            return 

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.run_pipeline()