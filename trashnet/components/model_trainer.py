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
                                       show_data_info,
                                       DataInspector)

from trashnet.utils.model_trainer_components_utils import *

class ModelTrainer:
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

                display_log_message(f"Loading label list from file: {color_text(self.data_transformation_config.label_list_path)}")
                label_list = load_object(file_path=self.data_transformation_config.label_list_path)

                display_log_message(f"Loading class weights from file: {color_text(self.data_transformation_config.class_weights_path)}")
                class_weights = load_object(file_path=self.data_transformation_config.class_weights_path)

                display_log_message(f"Initializing W&B")
                run = wandb.init(
                    project = self.wandb_config.project,
                    config = self.wandb_config.config
                )
                
                config = wandb.config


                display_log_message(f"Loading base model: {color_text(self.wandb_config.config['architecture'])}")
                base_model = tf.keras.applications.DenseNet121(
                    weights='imagenet', 
                    include_top=False,
                    input_shape=(*self.data_transformation_config.img_size, 3)
                )

                display_log_message(f"Building model")
                model = build_model(
                    input_shape=(*self.data_transformation_config.img_size, 3),
                    num_classes=label_list,
                    pretrained_model=base_model,
                )

                optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

                model.compile(
                    optimizer=optimizer,
                    loss=config.loss_function,
                    metrics=config.metrics
                )

                display_log_message(f"Model summary")
                model.summary()

                display_log_message(f"Loading train and validation data from TFRecord")
                train_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_transformation_config.train_tfrecord_data_path), compression="GZIP")
                valid_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_transformation_config.valid_tfrecord_data_path), compression="GZIP")

                display_log_message(f"Showing data info...")
                show_data_info(
                        train_dataset=train_tf_dataset_loaded,
                        valid_dataset=valid_tf_dataset_loaded,
                    )
                
                display_log_message(f"Inspecting data...")
                inspector = DataInspector(self.data_transformation_config.label_list)
                inspector.inspect(
                    train_dataset=train_tf_dataset_loaded,
                    valid_dataset=valid_tf_dataset_loaded
                )

                display_log_message(f"Loading train and validation data from TFRecord as batched")
                train_tf_images_batched = train_tf_dataset_loaded.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
                valid_tf_images_batched = valid_tf_dataset_loaded.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()

                display_info_dataset_batched(config.batch_size, train_tf_dataset_loaded, train_tf_images_batched, kind='train dataset')
                display_info_dataset_batched(config.batch_size, valid_tf_dataset_loaded, valid_tf_images_batched, kind='valid dataset')

                display_log_message(f"Training model...")
                training_logger = training_logger_callback(
                    log_file=self.model_trainer_config.training_tabel_path,
                    batch_size=config.batch_size,
                )
                plateau = plateau_callback()
                checkpoint = checkpoint_callback(
                    self.model_trainer_config.model_path
                )
                early_stopping = early_stopping_callback()
                wandb_logger = wandb_logger_callback(
                    validation_data=valid_tf_images_batched,
                    label_list=label_list,
                )

                history = model.fit(
                    train_tf_images_batched,
                    validation_data=valid_tf_images_batched,
                    epochs=config.epochs,
                    class_weight=class_weights,
                    callbacks=[
                        training_logger,
                        plateau,
                        checkpoint,
                        early_stopping,
                        wandb_logger,
                        WandbMetricsLogger(),
                    ]
                )

                wandb.finish()

                display_log_message(f"Saving tabel epoch to {color_text(self.model_trainer_config.tabel_epoch_path)}")
                history_df = pd.DataFrame(history.history).reset_index()
                history_df.rename(columns={'index': 'epoch'}, inplace=True)
                history_df['epoch'] = history_df['epoch'] + 1
                history_df = history_df.round({
                    'accuracy': 4,
                    'loss': 4,
                    'val_accuracy': 4,
                    'val_loss': 4,
                    'learning_rate': 10
                    }
                )

                tabel_epoch_dir_path = os.path.dirname(self.model_trainer_config.tabel_epoch_path)
                os.makedirs(tabel_epoch_dir_path, exist_ok=True)
                history_df.to_csv(self.model_trainer_config.tabel_epoch_path, index=False)

                display_log_message(f"Saveing plot training history")
                plot_training_history(history, self.model_trainer_config.plot_training_path)

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelTrainer')} class in {color_text(file_name_function)}")
                display_log_message(f"Model trainer config: {color_text(self.model_trainer_config)}")

                return self.model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
