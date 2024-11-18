import tensorflow as tf
import pandas as pd
import sys
import inspect 
import wandb

from wandb.integration.keras import WandbMetricsLogger
from trashnet.exception import TrashClassificationException
from trashnet.components.model_trainer_components.utils.prepare_callbacks import CallbacksManager
from trashnet.components.model_trainer_components.utils.display_info_dataset_batched import display_info_dataset_batched
from trashnet.components.model_trainer_components.utils.save_history_training import save_history_training
from trashnet.ml.model import build_model

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       load_object,
                                       show_data_info,
                                       DataInspector)


class ModelTrainer:
    def __init__(
            self, 
            data_preprocessing_config,
            model_trainer_config,
            wandb_config
        ):

        try:
            self.data_preprocessing_config = data_preprocessing_config
            self.model_trainer_config = model_trainer_config
            self.wandb_config = wandb_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_model_trainer(self):
        """
        Initiates the model training process.

        This method performs the following steps:
        1. Loads label list and class weights from files.
        2. Initializes Weights & Biases (W&B) for experiment tracking.
        3. Loads and builds the model.
        4. Compiles the model with specified optimizer, loss, and metrics.
        5. Loads and inspects train and validation data.
        6. Trains the model using the loaded data and saves training history.
        7. Saves the training history to a CSV file and plots it.
        8. Finishes the W&B run.

        Returns:
            ModelTrainerConfig: The configuration used for training the model.
        """
        try:
            # Log entry into method
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            # Load label list from file
            display_log_message(f"Loading label list from file: {color_text(self.data_preprocessing_config.labels_list_file_path)}")
            label_list = load_object(file_path=self.data_preprocessing_config.labels_list_file_path)

            # Load class weights from file
            display_log_message(f"Loading class weights from file: {color_text(self.data_preprocessing_config.class_weights_file_path)}")
            class_weights = load_object(file_path=self.data_preprocessing_config.class_weights_file_path)

            # Initialize W&B
            display_log_message(f"Initializing W&B")
            run = wandb.init(
                project=self.wandb_config.project_name,
                config=self.wandb_config.config
            )
            config = wandb.config

            # Load base model
            display_log_message(f"Loading base model: {color_text(self.wandb_config.config['architecture'])}")
            base_model = tf.keras.applications.DenseNet121(
                weights='imagenet', 
                include_top=False,
                input_shape=self.data_preprocessing_config.img_size
            )

            # Build model
            display_log_message(f"Building model")
            model = build_model(
                input_shape=self.data_preprocessing_config.img_size,
                num_classes=label_list,
                pretrained_model=base_model,
            )

            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            model.compile(
                optimizer=optimizer,
                loss=config.loss_function,
                metrics=config.metrics
            )

            # Display model summary
            display_log_message(f"Model summary")
            model.summary()

            # Load data from TFRecord
            display_log_message(f"Loading train and validation data from TFRecord")
            train_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_preprocessing_config.train_tfrecord_file_path), compression="GZIP")
            valid_tf_dataset_loaded = tf.data.Dataset.load(str(self.data_preprocessing_config.valid_tfrecord_file_path), compression="GZIP")

            # Show data info
            display_log_message(f"Showing data info...")
            show_data_info(
                train_dataset=train_tf_dataset_loaded,
                valid_dataset=valid_tf_dataset_loaded,
            )

            # Inspect data
            display_log_message(f"Inspecting data...")
            inspector = DataInspector(self.data_preprocessing_config.label_list)
            inspector.inspect(
                train_dataset=train_tf_dataset_loaded,
                valid_dataset=valid_tf_dataset_loaded
            )

            # Load data from TFRecord as batched
            display_log_message(f"Loading train and validation data from TFRecord as batched")
            train_tf_images_batched = train_tf_dataset_loaded.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
            valid_tf_images_batched = valid_tf_dataset_loaded.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()

            # Display data info as batched
            display_info_dataset_batched(config.batch_size, train_tf_dataset_loaded, train_tf_images_batched, kind='train dataset')
            display_info_dataset_batched(config.batch_size, valid_tf_dataset_loaded, valid_tf_images_batched, kind='valid dataset')

            # Train model
            display_log_message(f"Training model...")
            callbacks_manager = CallbacksManager(
                log_file=self.model_trainer_config.training_table_file_path,
                batch_size=config.batch_size,
                model_path=self.model_trainer_config.model_file_path,
                validation_data=valid_tf_images_batched,
                label_list=label_list
            )
            callbacks = callbacks_manager.get_callbacks()

            history = model.fit(
                train_tf_images_batched,
                validation_data=valid_tf_images_batched,
                epochs=config.epochs,
                class_weight=class_weights,
                callbacks=[
                    *callbacks,
                    WandbMetricsLogger(),
                ]
            )

            # Save table of epoch to file
            display_log_message(f"Saving table epoch to {color_text(self.model_trainer_config.epoch_table_file_path)}")
            history_df = pd.DataFrame(history.history).reset_index()
            history_df.rename(columns={'index': 'epoch'}, inplace=True)
            history_df['epoch'] = history_df['epoch'] + 1
            history_df = history_df.round({
                'accuracy': 4,
                'loss': 4,
                'val_accuracy': 4,
                'val_loss': 4,
                'learning_rate': 10
            })
            history_df.to_csv(self.model_trainer_config.epoch_table_file_path, index=False)

            # Finish W&B
            # wandb.finish()

            # Save plot of training history
            display_log_message(f"Saving plot training history")
            save_history_training(history, self.model_trainer_config.training_plot_file_path)

            # Display exit message
            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            # Display model trainer config
            display_log_message(f"{class_name} config: {color_text(self.model_trainer_config)}")

            return self.model_trainer_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
