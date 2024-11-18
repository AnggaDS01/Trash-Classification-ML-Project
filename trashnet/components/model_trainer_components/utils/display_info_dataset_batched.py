import sys
import re
import tensorflow as tf

from trashnet.utils.main_utils import custom_title_print
from trashnet.exception import TrashClassificationException

def display_info_dataset_batched(
        batch_size: int = 32, 
        dataset: tf.data.Dataset = None, 
        dataset_batched: tf.data.Dataset = None, 
        kind: str = 'train' 
    ) -> None:

    """
    Display information about the dataset and batched dataset.

    Args:
        batch_size (int): The batch size of the dataset.
        dataset (tf.data.Dataset): The dataset to be displayed.
        dataset_batched (tf.data.Dataset): The batched dataset to be displayed.
        kind (str): The type of dataset being displayed (e.g. 'train', 'test', 'validation').
    """
    try:
        custom_title_print(f' {kind} ')
        print(f"Info data: {dataset_batched}")
        print(f"Number of data: {len(dataset)}")
        if not re.search('test', kind.lower(), re.IGNORECASE):
            # If the dataset is not the test dataset, then display the batch size and the number of data
            print(f"AFTER BATCH: {batch_size}")
            print(f"Number of data: {len(dataset_batched)}")

    except Exception as e:
        raise TrashClassificationException(e, sys)