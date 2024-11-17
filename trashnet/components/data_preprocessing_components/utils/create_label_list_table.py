import sys
import tensorflow as tf

from trashnet.exception import TrashClassificationException

def create_label_list_table(
        label_list: list=[], 
        default_value: int=-1
    ) -> tf.lookup.StaticHashTable:
    """
    Creates a StaticHashTable for encoding labels based on the given list of labels.
    Args:
    - label_list (list): List of labels to encode.
    - default_value (int): The default value if the label is not found in the table.
    Returns:
    - label_table (tf.lookup.StaticHashTable): The hash table for encoding labels.
    """

    try:

        # Create a tensor from a list of labels (keys)
        keys_tensor = tf.constant(label_list)

        # Create integer values associated with each label (values)
        values_tensor = tf.range(len(label_list))

        # Initialize key-value pairs for the hash table
        table_init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)

        # Creating a StaticHashTable
        label_table = tf.lookup.StaticHashTable(table_init, default_value=default_value)

        return label_table
    
    except Exception as e:
        raise TrashClassificationException(e, sys)