import sys
import numpy as np
import tensorflow as tf

from trashnet.exception import TrashClassificationException

def predict_image(
        model: tf.keras.Model = None, 
        processed_image: np.ndarray = None, 
        label_list: list = None, 
        threshold: float = 0.5
    ) -> str:
    """
    Predicts the label of a processed image using a trained model.

    Args:
    - model (tf.keras.Model): The trained model used for prediction.
    - processed_image (np.ndarray): The preprocessed image to be classified.
    - label_list (list): List of possible labels for classification.
    - threshold (float): Threshold for binary classification.

    Returns:
    - str: The predicted label for the input image.
    """
    try:
        # Make predictions on the processed image
        predictions = model.predict(processed_image, verbose=0)
        predictions = tf.squeeze(predictions).numpy()

        # Determine the predicted label based on the number of labels
        if len(label_list) == 2:
            # Binary classification
            predicted_label = label_list[int(predictions > threshold)]
        else:
            # Multi-class classification
            predicted_label = label_list[np.argmax(predictions)]

        return predicted_label

    except Exception as e:
        # Raise a custom exception if an error occurs
        raise TrashClassificationException(e, sys)