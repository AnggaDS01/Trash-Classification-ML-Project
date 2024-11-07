import sys
import inspect
from trashnet.exception import TrashClassificationException
from trashnet.components.hyperparameter_tuning import HyperParameterTuning

from trashnet.configuration.configuration import ConfigurationManager


from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text)

class HyperparameterTuningPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_transformation_config = config.get_data_transformation_config()
        self.model_trainer_config = config.get_model_trainer_config()
        self.wandb_config = config.get_wandb_config()
    
    def start_hyperparameter_tuning(
            self,
        ):
        try:
            function_name, file_name_function = display_function_name(inspect.currentframe())
            display_log_message(f"Started the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}")

            hyperparameter_tuning = HyperParameterTuning(
                data_transformation_config = self.data_transformation_config,
                model_trainer_config = self.model_trainer_config,
                wandb_config = self.wandb_config,
            )

            hyperparameter_tuning.initiate_hyperparaeter_tuning()

            display_log_message(f"Exited the {color_text(function_name)} method of {color_text('TrainPipeline')} class in {color_text(file_name_function)}\n\n")

            return

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
        

    def run_pipeline(self) -> None:
        try:
            self.start_hyperparameter_tuning()
        except Exception as e:
            raise TrashClassificationException(e, sys)