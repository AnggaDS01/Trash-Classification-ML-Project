import tensorflow as tf
import sys
import inspect 
import wandb

from trashnet.exception import TrashClassificationException
from trashnet.components.hyperparameter_tuning_components.utils.sweep_setup import train
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       load_object)


class HyperParameterTuning:
    def __init__(
            self, 
            data_preprocessing_config,
            model_trainer_config,
            wandb_config,
        ):

        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.model_trainer_config = model_trainer_config
            self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)


    def initiate_hyperparaeter_tuning(self):
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
            display_log_message(f"Loading label list from file: {color_text(self.data_preprocessing_config.labels_list_file_path)}")
            label_list = load_object(file_path=self.data_preprocessing_config.labels_list_file_path)

            display_log_message(f"Loading class weights from file: {color_text(self.data_preprocessing_config.class_weights_file_path)}")
            class_weights = load_object(file_path=self.data_preprocessing_config.class_weights_file_path)

            display_log_message(f"Loading base model: {color_text(self.wandb_config.config['architecture'])}")
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet', 
                include_top=False,
                input_shape=self.data_preprocessing_config.img_size
            )

            display_log_message(f"Loading train and validation data from TFRecord")
            train_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_preprocessing_config.train_tfrecord_file_path), compression="GZIP")
            valid_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_preprocessing_config.valid_tfrecord_file_path), compression="GZIP")

            sweep_id = wandb.sweep(
                sweep=self.wandb_config.sweep_config, 
                project=self.wandb_config.project_name
            )

            wandb.agent(
                sweep_id=sweep_id, 
                function=lambda: train(
                    base_model=base_model,
                    train_tf_dataset=train_tf_dataset_loaded,
                    valid_tf_dataset=valid_tf_dataset_loaded,
                    image_size=self.data_preprocessing_config.img_size,
                    label_list=label_list,
                    loss_function=self.model_trainer_config.loss_function,
                    metrics=self.model_trainer_config.metrics,
                    class_weights=class_weights,
                ), 
                count=self.wandb_config.sweep_count
            )

            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            return 

        except Exception as e:
            raise TrashClassificationException(e, sys)
