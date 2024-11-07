import sys
import inspect 
import tensorflow as tf

from trashnet.exception import TrashClassificationException
from trashnet.ml.model import build_model
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       load_object,
                                       paths_exist,
                                       show_data_info,
                                       DataInspector)

from trashnet.utils.model_trainer_components_utils import *

class ModelEvaluation:
    def __init__(
            self, 
            data_transformation_config,
            model_trainer_config,
            wandb_config
        ):

        try:
            self.data_transformation_config = data_transformation_config
            self.model_trainer_config = model_trainer_config
            self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_model_trainer(self):
        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")

        try:

            required_files = [
                self.model_trainer_config.model_path,
            ]

            if paths_exist(required_files):
                display_log_message("All required files and folders are present. Skipping the process.")

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")

                display_log_message(f"Model trainer config: {color_text(self.model_trainer_config)}")

                return self.model_trainer_config

            else:
                display_log_message("One or more required files/folders are missing. Running the process...")





                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")
                display_log_message(f"Model trainer config: {color_text(self.model_trainer_config)}")

                return self.model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
