import sys
import collections
from trashnet.exception import TrashClassificationException

def print_class_distribution(
        distribution: collections.Counter=None
    ) -> None:

    """
    Prints the class distribution to the screen.
    Args:
        distribution(collections.Counter): A Counter object containing the class distribution.
    """
    try:
        for class_name, count in sorted(distribution.items()):
            print(f"{class_name}: {count}")

    except Exception as e:
        raise TrashClassificationException(e, sys)