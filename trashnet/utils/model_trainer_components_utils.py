import tensorflow as tf
import numpy as np
import seaborn as sns
import re
import os
import csv
import matplotlib.pyplot as plt
import wandb

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from trashnet.utils.main_utils import custom_title_print

def display_info_dataset_batched(batch_size, dataset, dataset_batched, kind):
    custom_title_print(f' {kind} ')
    print(f"Info data: {dataset_batched}")
    print(f"Number of data: {len(dataset)}")
    if not re.search('test', kind.lower(), re.IGNORECASE):
        print(f"AFTER BATCH: {batch_size}")
        print(f"Number of data: {len(dataset_batched)}")


def training_logger_callback(log_file, batch_size):
    training_logger = TrainingLogger(
        log_file = log_file,  
        batch_size = batch_size
    )
    return training_logger  


def plateau_callback():
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',  
        factor=0.5,  
        patience=10, 
        verbose=1, 
        mode='auto', 
        min_delta=0.001,
        cooldown=0,
        min_lr=0 
    )
    return plateau_callback  


def checkpoint_callback(model_path):
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = model_path, 
        monitor = 'val_loss', 
        save_best_only = True,
        save_weights_only = False, 
        mode = 'min',
        verbose = 1
    )
    return checkpoint_callback  


def early_stopping_callback():
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=25, 
        restore_best_weights=True, 
        verbose=1  
    )
    return early_stopping  


def wandb_logger_callback(validation_data, label_list):
    wandb_logger = WandbImageLogger(
        validation_data=validation_data,  
        label_list=label_list,  
        sample_count=10 
    )
    return wandb_logger  

def plot_training_history(history, save_path):
    """
    Plot training and validation accuracy and loss curves.

    Args:
    - history: History object yang dikembalikan oleh model.fit()
    - save_path (str): Path di mana gambar kurva akan disimpan.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    # Plot akurasi
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Buat direktori jika belum ada, lalu simpan plot
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(save_path)


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
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
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
        # Ambil batch pertama dari data validasi
        images, labels = next(iter(self.validation_data))

        # Ambil sejumlah sampel prediksi untuk logging
        sample_images = images[:self.sample_count]
        sample_labels = labels[:self.sample_count]
        predictions = self.model.predict(sample_images)

        wandb_images = []
        for i in range(self.sample_count):
            true_label = self.label_list[sample_labels[i].numpy()]
            predicted_label = self.label_list[np.argmax(predictions[i])]

            plt.figure()
            plt.imshow(sample_images[i])
            plt.title(f"True: {true_label}, Pred: {predicted_label}")

            # Simpan gambar ke wandb
            wandb_images.append(wandb.Image(plt, caption=f"True: {true_label}, Pred: {predicted_label}"))
            plt.close()

        # Log gambar-gambar ke wandb
        wandb.log({"predictions": wandb_images}, step=epoch)