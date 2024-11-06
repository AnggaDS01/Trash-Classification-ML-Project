import sys
import inspect 
import re 
import random 
import os
import tensorflow as tf
import numpy as np
import wandb

from trashnet.constant.wandb_config import WANDB_CONFIG
from trashnet.exception import TrashClassificationException
from trashnet.entity.config_entity import ModelTrainerConfig
from trashnet.entity.artifacts_entity import (ModelTrainerArtifact,
                                              DataTransformationArtifact)

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       custom_title_print,
                                       paths_exist)

class ModelTrainer:
    def __init__(
            self, 
            data_transformation_artifact: DataTransformationArtifact,
            model_trainer_config: ModelTrainerConfig
        ):

        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
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

                model_trainer_artifact = ModelTrainerArtifact(
                    model_path = self.model_trainer_config.model_path,
                )

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")
                display_log_message(f"Model trainer artifact: {color_text(model_trainer_artifact)}")

                return model_trainer_artifact

            else:
                display_log_message("One or more required files/folders are missing. Running the process...")

                # Inisialisasi W&B menggunakan konfigurasi dari file terpisah
                run = wandb.init(
                    project=WANDB_CONFIG["project"],
                    config=WANDB_CONFIG["config"]
                )
                
                config = wandb.config

                print("oke")

                model_trainer_artifact = ModelTrainerArtifact(
                    model_path = self.model_trainer_config.model_path,
                )

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")
                display_log_message(f"Model trainer artifact: {color_text(model_trainer_artifact)}")

                return model_trainer_artifact

        except Exception as e:
            raise TrashClassificationException(e, sys)
