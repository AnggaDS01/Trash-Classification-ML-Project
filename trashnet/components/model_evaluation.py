import sys
import inspect 
import tensorflow as tf
import wandb

from trashnet.exception import TrashClassificationException
from trashnet.utils.model_evaluation_components_utils import *
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       load_object,
                                       paths_exist)

from trashnet.utils.model_trainer_components_utils import *

class ModelEvaluation:
    def __init__(
            self,
            data_transformation_config, 
            model_trainer_config,
            model_evaluation_config,
            # wandb_config
        ):

        try:
            self.data_transformation_config = data_transformation_config
            self.model_trainer_config = model_trainer_config
            self.model_evaluation_config = model_evaluation_config
            # self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_model_trainer(self):
        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('ModelEvaluation')} class in {color_text(file_name_function)}")

        try:

            required_files = [
                self.model_evaluation_config.plot_confusion_matrix_path,
                self.model_evaluation_config.classification_report_path
            ]

            if paths_exist(required_files):
                display_log_message("All required files and folders are present. Skipping the process.")

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelEvaluation')} class in {color_text(file_name_function)}")

                display_log_message(f"Model Evaluation config: {color_text(self.model_evaluation_config)}")

                return self.model_evaluation_config

            else:
                display_log_message("One or more required files/folders are missing. Running the process...")

                display_log_message(f"Loading label list from file: {color_text(self.data_transformation_config.label_list_path)}")
                label_list = load_object(file_path=self.data_transformation_config.label_list_path)

                display_log_message(f"Loading the best model from {color_text(self.model_trainer_config.model_path)}")
                best_model = tf.keras.models.load_model(self.model_trainer_config.model_path)

                display_log_message(f"Evaluating the model on validation data..")
                valid_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_transformation_config.valid_tfrecord_data_path), compression="GZIP")
                valid_tf_images_batched = valid_tf_dataset_loaded.batch(self.model_trainer_config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
                evaluation = best_model.evaluate(valid_tf_images_batched)
                print(f'Model evaluation on validation data: {evaluation}')

                display_log_message("Saving confusion matrix and classification report")

                # wandb.init(
                #     project = self.wandb_config.project,
                #     config = self.wandb_config.config
                # )

                evaluate_model(
                    best_model,
                    valid_tf_images_batched,
                    label_list,
                    confusion_plot_path=self.model_evaluation_config.plot_confusion_matrix_path,
                    classification_report_path=self.model_evaluation_config.classification_report_path,
                    save_plot=True,
                    save_report=True,
                    normalize=self.model_evaluation_config.normalize,
                    figsize=self.model_evaluation_config.figsize
                )

                wandb.finish()

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelEvaluation')} class in {color_text(file_name_function)}")
                display_log_message(f"Model Evaluation config: {color_text(self.model_evaluation_config)}")

                return self.model_evaluation_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
