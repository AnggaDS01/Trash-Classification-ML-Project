import sys
import wandb
import tensorflow as tf

from trashnet.ml.model import build_model
from trashnet.exception import TrashClassificationException


def train(
        base_model: tf.keras.Model=None,
        train_tf_dataset: tf.data.Dataset=None,
        valid_tf_dataset: tf.data.Dataset=None,
        image_size: tuple=None,
        label_list: int=None,
        loss_function: str=None,
        metrics: list=None,
        class_weights: dict=None,
        config: dict=None
    ) -> None:

    try:
        with wandb.init(config=config):
            config = wandb.config

            # Build and compile the model with sweep hyperparameters
            model = build_model(
                input_shape=image_size,
                num_classes=label_list,
                pretrained_model=base_model,
            )

            train_tf_images_batched = train_tf_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()
            valid_tf_images_batched = valid_tf_dataset.batch(config.batch_size).prefetch(tf.data.AUTOTUNE).cache()

            optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)
            model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)

            # Train the model with the sweep configurations
            model.fit(
                train_tf_images_batched,
                validation_data=valid_tf_images_batched,
                epochs=config.epochs,
                class_weight=class_weights,
                callbacks=[
                    tf.keras.callbacks.LambdaCallback(
                        on_epoch_end=lambda epoch, logs: wandb.log({"val_accuracy": logs["val_accuracy"], "train_accuracy": logs["accuracy"]})
                    )
                ]
            )

        wandb.finish()
        
    except Exception as e:
        raise TrashClassificationException(e, sys)