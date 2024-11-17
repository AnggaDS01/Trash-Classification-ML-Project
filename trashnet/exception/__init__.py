import sys
import logging

def error_message_detail(error, error_detail: sys):
    """
    Formats the error message with file name and line number.
    
    Args:
        error (Exception): The exception that occurred.
        error_detail (sys): The sys module to retrieve traceback.
    
    Returns:
        str: Formatted error message.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = "Error occurred in script [{0}] line [{1}] message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class TrashClassificationException(Exception):
    def __init__(self, error_message, error_detail):
        """
        Custom Exception class to log error details to log file.

        Args:
            error_message (str): Error message.
            error_detail (sys): System info to extract traceback.
        """
        super().__init__(error_message)
        
        # Format and save error message with traceback details
        self.error_message = error_message_detail(error_message, error_detail)
        
        # Log the error message
        logging.error(self.error_message)

    def __str__(self):
        return self.error_message