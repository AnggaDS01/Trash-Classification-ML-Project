import sys
import matplotlib.pyplot as plt

from trashnet.exception import TrashClassificationException

def save_history_training(
        history: dict = None, 
        save_path: str = None
    ) -> None:

    """
    Plot training and validation accuracy and loss curves.

    Args:
    - history (dict): History object returned by model.fit()
    - save_path (str): Path to save the plot

    """

    try:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        # Plot accuracy
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

        plt.savefig(save_path)

    except Exception as e:
        raise TrashClassificationException(e, sys)