import sys
import os
import tensorflow as tf
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from trashnet.exception import TrashClassificationException

def calculate_class_distribution_tf(
        dataset: tf.data.Dataset=None, 
        class_labels: list=None
    ) -> tuple:
    
    """
    Compute the class distribution and class weight using `compute_class_weight` from sklearn.
    Args:
        dataset (tf.data.Dataset): TensorFlow dataset containing images and labels.
        class_labels (list): An ordered list of class names (e.g. ['cardboard', 'glass', ...]).
    Returns:
        tuple: class_counts (Counter), class_weights (dict)
            class_weights in the form {label_index: weight}
    """
    
    try:
        # Takes the class name from the path and converts it to a list
        class_names = dataset.map(lambda x: extract_class_from_path_tf(x))
        class_names_list = list(class_names.batch(1000).as_numpy_iterator())
        all_class_names = [name.decode('utf-8') for batch in class_names_list for name in batch]

        # Calculate class distribution
        class_counts = Counter(all_class_names)

        # Convert class name to numeric index as per `class_labels`
        class_indices = [class_labels.index(name) for name in all_class_names]

        # Calculate class weight using sklearn
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_indices),
            y=class_indices
        )

        # Create class_weights dict according to the index
        class_weights = {i: weight for i, weight in enumerate(class_weight_values)}

        return class_counts, class_weights

    except Exception as e:
        raise TrashClassificationException(e, sys)


def extract_class_from_path_tf(
        path_tensor: tf.Tensor
    ) -> tf.Tensor:

    """
    Extracts the class name from the file path stored in tensor form.
    Args:
        path_tensor(tf.Tensor): The tensor containing the file path.
    Returns:
        tf.Tensor: A tensor containing the name of the class extracted from the file path.
    """

    try:
        parts = tf.strings.split(path_tensor, os.path.sep)
        class_name = parts[-2]  # The class name is usually in the penultimate position in the line
        return class_name
    
    except Exception as e:
        raise TrashClassificationException(e, sys)