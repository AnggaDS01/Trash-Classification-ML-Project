import sys
import inspect
from trashnet.exception import TrashClassificationException
from trashnet.components.hyperparameter_tuning_components.hyperparameter_tuning import HyperParameterTuning

from trashnet.configuration.configuration import ConfigurationManager


from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text)

class HyperparameterTuningPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        self.model_trainer_config = config.get_model_trainer_config()
        self.wandb_config = config.get_wandb_config()
    
    def start_hyperparameter_tuning(
            self,
        ):
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            hyperparameter_tuning = HyperParameterTuning(
                data_preprocessing_config = self.data_preprocessing_config,
                model_trainer_config = self.model_trainer_config,
                wandb_config = self.wandb_config,
            )

            hyperparameter_tuning.initiate_hyperparaeter_tuning()

            # Display exit message
            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            return

        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def run_pipeline(self) -> None:
        try:
            self.start_hyperparameter_tuning()
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
if __name__ == "__main__":
    pipeline = HyperparameterTuningPipeline()
    pipeline.run_pipeline()