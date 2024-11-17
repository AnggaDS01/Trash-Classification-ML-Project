import os
import sys
import zipfile

from trashnet.utils.main_utils import display_log_message
from trashnet.exception import TrashClassificationException
from pathlib import Path

def extract_zip(
    zip_file_path: Path, 
    extract_dir: Path, 
    is_file_removed: bool = True
) -> None:
    """
    Extracts a zip file to a specified directory.

    Args:
        zip_file_path (Path): The path to the zip file.
        extract_dir (Path): The directory where files will be extracted.
        is_file_removed (bool): Delete the zip file after extraction if True.
    
    Raises:
        zipfile.BadZipFile: If the file is not a valid zip file.
    """
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            display_log_message(f"Files extracted to {extract_dir}", is_log=False)

        # Remove zip file if specified
        if is_file_removed and os.path.exists(zip_file_path):
            os.remove(zip_file_path)
            display_log_message("Downloaded zip file removed.", is_log=False)

    except zipfile.BadZipFile:
        raise Exception("Error: The downloaded file is not a valid zip file.")
    
    except Exception as e:
        raise TrashClassificationException(e, sys)