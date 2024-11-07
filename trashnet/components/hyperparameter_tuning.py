import tensorflow as tf
import pandas as pd
import sys
import inspect 
import wandb

from wandb.integration.keras import WandbMetricsLogger
from trashnet.exception import TrashClassificationException
from trashnet.ml.model import build_model
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       load_object,
                                       paths_exist,
                                       show_data_info)

from trashnet.utils.model_trainer_components_utils import *

class HyperParameterTuning:
    def __init__(
            self, 
            data_transformation_config,
            wandb_config,
        ):

        try:
            self.data_transformation_config = data_transformation_config
            self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
    
    def train(
            self, 
            base_model,
            train_data,
            val_data,
            image_size,
            label_list,
            loss_function,
            metrics,
            class_weights=None,
            config=None
        ):
        with wandb.init(config=config):
            config = wandb.config

            # Build and compile the model with sweep hyperparameters
            model = build_model(
                input_shape=(*image_size, 3),
                num_classes=label_list,
                pretrained_model=base_model,
            )

            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            # Train the model with the sweep configurations
            model.fit(
                train_data,
                validation_data=val_data,
                epochs=config.epochs,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: wandb.log({"val_accuracy": logs["val_accuracy"], "train_accuracy": logs["accuracy"]})
                    )
                ]
            )


    def initiate_hyperparaeter_tuning(self):

        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('HyperParameterTuning')} class in {color_text(file_name_function)}")

        try:
            display_log_message(f"Loading label list from file: {color_text(self.data_transformation_config.label_list_path)}")
            label_list = load_object(file_path=self.data_transformation_config.label_list_path)

            display_log_message(f"Loading class weights from file: {color_text(self.data_transformation_config.class_weights_path)}")
            class_weights = load_object(file_path=self.data_transformation_config.class_weights_path)

            print('oke')


            # sweep_id = wandb.sweep(
            #     sweep=self.wandb_config.sweep_config, 
            #     project=self.wandb_config.project
            # )

            # wandb.agent(
            #     sweep_id=sweep_id, 
            #     function=lambda: self.train(
            #         base_model=self.model_trainer_config.base_model,
            #         train_data=self.data_transformation_config.train_tfrecord_data_path,
            #         val_data=self.data_transformation_config.valid_tfrecord_data_path,
            #         image_size=self.data_transformation_config.image_size,
            #         label_list=label_list,
            #         loss_function=self.model_trainer_config.loss_function,
            #         metrics=self.model_trainer_config.metrics,
            #         class_weights=class_weights,
            #     ), 
            #     count=self.wandb_config.sweep_count
            # )

            display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")

            return 

        except Exception as e:
            raise TrashClassificationException(e, sys)
