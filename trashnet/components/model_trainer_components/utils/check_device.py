import tensorflow as tf
import sys

from trashnet.exception import TrashClassificationException
from trashnet.utils.main_utils import display_log_message, color_text

def check_device() -> str:
    """
    Checks whether a GPU is available and sets memory growth.
    Returns the device used for training.
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            display_log_message("GPU detected. Training will run on GPU.")
            display_log_message(f"GPU devices: {color_text(gpus)}")
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                display_log_message("Memory growth enabled for GPUs.")
            except RuntimeError as e:
                display_log_message(f"Memory growth failed: {e}")
        else:
            display_log_message("No GPU detected. Training will run on CPU.")

        # Determine the device to use for training
        device = "GPU" if tf.test.is_gpu_available() else "CPU"
        display_log_message(f"Training will proceed on: {color_text(device)}")
        return device

    except Exception as e:
        raise TrashClassificationException(e, sys)