import requests
import sys
import logging

from trashnet.utils.main_utils import display_log_message
from trashnet.exception import TrashClassificationException
from tqdm import tqdm
from pathlib import Path

def download_zip(
    url: str, 
    save_zip_file_path: Path, 
    chunk_size: int = 1024
) -> None:
    """
    Downloads a file from a given URL to the specified path.
    
    Args:
        url (str): URL of the file to download.
        save_zip_file_path (Path): The path where the file will be saved.
        chunk_size (int): The chunk size for download. Default is 1024 (1 KB).
    
    Raises:
        requests.exceptions.RequestException: If an error occurs during download.
    """
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()  # Check for HTTP errors
        
        total_size = int(response.headers.get('content-length', 0))
        with open(save_zip_file_path, "wb") as file, tqdm(
                desc=f"Downloading {save_zip_file_path}",
                total=total_size,
                unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                file.write(chunk)
                bar.update(len(chunk))

        display_log_message(f"File downloaded to {save_zip_file_path}", is_log=False)

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading the file: {e}")
    
    except Exception as e:
        logging.info(e)
        raise TrashClassificationException(e, sys)