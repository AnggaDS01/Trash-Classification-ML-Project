import sys
import inspect 
import tensorflow as tf
import wandb

from trashnet.exception import TrashClassificationException
from trashnet.components.model_evaluation_components.utils.evaluate_model import evaluate_model
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       load_object,
                                       custom_title_print)


class ModelEvaluation:
    def __init__(
            self,
            data_preprocessing_config, 
            model_trainer_config,
            model_evaluation_config,
            # wandb_config
        ):

        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.model_trainer_config = model_trainer_config
            self.model_evaluation_config = model_evaluation_config
            # self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_model_evaluation(self):

        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            custom_title_print(f'{class_name}')
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            display_log_message(f"Loading label list from file: {color_text(self.data_preprocessing_config.labels_list_file_path)}")
            label_list = load_object(file_path=self.data_preprocessing_config.labels_list_file_path)

            display_log_message(f"Loading the best model from {color_text(self.model_trainer_config.model_file_path)}")
            best_model = tf.keras.models.load_model(self.model_trainer_config.model_file_path)

            display_log_message(f"Evaluating the model on validation data..")
            valid_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_preprocessing_config.valid_tfrecord_file_path), compression="GZIP")
            valid_tf_images_batched = valid_tf_dataset_loaded.batch(self.model_trainer_config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
            # evaluation = best_model.evaluate(valid_tf_images_batched)
            # print(f'Model evaluation on validation data: {evaluation}')

            display_log_message("Saving confusion matrix and classification report")

            # wandb.init(
            #     project = self.wandb_config.project,
            #     config = self.wandb_config.config
            # )

            evaluate_model(
                model = best_model,
                tf_dataset = valid_tf_images_batched,
                class_names = label_list,
                confusion_plot_path = self.model_evaluation_config.plot_confusion_matrix_file_path,
                classification_report_path = self.model_evaluation_config.classification_report_file_path,
                save_plot = True,
                save_report = True,
                normalize = self.model_evaluation_config.normalize_confusion_matrix,
                figsize = self.model_evaluation_config.figsize
            )

            wandb.finish()

            # Display exit message
            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            # Display model trainer config
            display_log_message(f"{class_name} config: {color_text(self.model_evaluation_config)}")

            return self.model_evaluation_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
