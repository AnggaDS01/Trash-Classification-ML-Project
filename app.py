from trashnet.logger import logging
from trashnet.exception import TrashClassificationException
import sys

def test_logging():
    logging.info("Test logging")

def test_exception():
    try:
        a = 1/0
    except Exception as e:
        raise TrashClassificationException(e, sys)
    

if __name__ == "__main__":
    test_logging()
    test_exception()