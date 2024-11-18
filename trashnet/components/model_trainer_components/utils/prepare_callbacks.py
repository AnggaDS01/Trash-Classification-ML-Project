import os
import csv
import numpy as np
import wandb
import matplotlib.pyplot as plt
import tensorflow as tf

class CallbacksManager:
    def __init__(
            self, 
            log_file: str = None, 
            batch_size: int = None, 
            model_path: str = None, 
            validation_data: tf.data.Dataset = None, 
            label_list: list = None
        ):

        self.log_file = log_file
        self.batch_size = batch_size
        self.model_path = model_path
        self.validation_data = validation_data
        self.label_list = label_list

    def training_logger_callback(self):
        """
        Initializes the TrainingLogger callback.
        
        Returns:
        - TrainingLogger: A custom logger for tracking training progress.
        """
        return TrainingLogger(
            log_file=self.log_file,
            batch_size=self.batch_size
        )

    def plateau_callback(self):
        """
        Initializes the ReduceLROnPlateau callback.
        
        Returns:
        - tf.keras.callbacks.ReduceLROnPlateau: A callback to reduce the learning rate when the validation loss plateaus.
        """
        return tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            verbose=1,
            mode='auto',
            min_delta=0.001,
            cooldown=0,
            min_lr=0
        )

    def checkpoint_callback(self):
        """
        Initializes the ModelCheckpoint callback.
        
        Returns:
        - tf.keras.callbacks.ModelCheckpoint: A callback to save the best model weights based on validation loss.
        """
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )

    def early_stopping_callback(self):
        """
        Initializes the EarlyStopping callback.
        
        Returns:
        - tf.keras.callbacks.EarlyStopping: A callback to stop training early when validation loss stops improving.
        """

        return tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            verbose=1
        )

    def wandb_logger_callback(self):
        """
        Initializes the WandbImageLogger callback.
        
        Returns:
        - WandbImageLogger: A custom logger for logging images to Weights and Biases (WandB).
        """

        return WandbImageLogger(
            validation_data=self.validation_data,
            label_list=self.label_list,
            sample_count=10
        )

    def get_callbacks(self):
        """
        Returns an array of all callbacks.
        
        Returns:
        - list: A list of initialized callbacks based on the available parameters.
        """
        callbacks = [
            self.wandb_logger_callback(),
            self.training_logger_callback(),
            self.plateau_callback(),
            self.checkpoint_callback(),
            self.early_stopping_callback(),
        ]
        return callbacks



class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Callback untuk mencatat hasil pelatihan pada setiap epoch ke dalam file CSV.

    Args:
        log_file (str): Path ke file log CSV.
        batch_size (int): Ukuran batch yang digunakan dalam pelatihan.
    """
    def __init__(self, log_file, batch_size):
        super(TrainingLogger, self).__init__()
        self.log_file = log_file
        self.batch_size = batch_size

        # Membuat file CSV dan menulis header-nya jika belum ada
        if not os.path.exists(log_file):
            with open(log_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Epoch", "Batch Size", "val_acc (%)", "val_loss (%)"])

    def on_epoch_end(self, epoch, logs=None):
        """
        Callback yang dipanggil pada akhir setiap epoch untuk mencatat
        val_loss dan val_accuracy ke dalam file CSV.

        Args:
            epoch (int): Indeks epoch yang sedang berlangsung.
            logs (dict): Dictionary yang menyimpan metrik pelatihan seperti val_loss dan val_accuracy.
        """
        val_loss = logs.get('val_loss', 0) * 100  
        val_acc = logs.get('val_accuracy', 0) * 100  


        # Menulis hasil ke CSV
        with open(self.log_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch + 1, self.batch_size, round(val_acc, 4), round(val_loss, 4)])


# Callback untuk logging gambar prediksi
class WandbImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data, label_list, sample_count=5):
        super().__init__()
        self.validation_data = validation_data
        self.label_list = label_list
        self.sample_count = sample_count

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve a batch of images and labels from the validation dataset
        images, labels = next(iter(self.validation_data))
        total_samples = tf.shape(images)[0]

        # Randomly select a subset of samples
        random_indices = tf.random.shuffle(tf.range(total_samples))[:self.sample_count]
        random_images = tf.gather(images, random_indices)
        random_labels = tf.gather(labels, random_indices)

        # Predict the labels for the randomly selected images
        predictions = self.model.predict(random_images)

        wandb_images = []
        for i in range(self.sample_count):
            # Get the true and predicted labels for each image
            true_label = self.label_list[random_labels[i].numpy()]
            predicted_label = self.label_list[np.argmax(predictions[i])]

            # Plot the image with the true and predicted labels as title
            plt.figure()
            plt.imshow(random_images[i].numpy())

            # Append the image to the list with WandB caption
            wandb_images.append(wandb.Image(plt, caption=f"True: {true_label}, Pred: {predicted_label}"))
            plt.close()

        # Log the images to WandB under the key 'predictions'
        wandb.log({"predictions": wandb_images}, step=epoch)