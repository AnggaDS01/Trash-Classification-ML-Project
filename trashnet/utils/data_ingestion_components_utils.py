import sys 
import os 
import zipfile 
import requests
from tqdm import tqdm
from pathlib import Path

from trashnet.utils.main_utils import (display_log_message, 
                                       color_text)

def download_and_extract_zip(
    url: str=None, 
    save_dir_path: Path=None, 
    extract_dir_path: Path=None, 
    filename: str="dataset.zip", 
    chunk_size: int=1024, 
    is_file_removed: bool=True
) -> None:
    """
    Downloads a zip file from a URL and extracts it to the specified directory only if there is no extract result yet.
    Args:
        url(str): URL of the zip file to download.
        save_dir_path (str): The directory where to save the zip file.
        extract_dir_path (str): The destination directory to extract the zip file.
        filename (str, optional): The name of the saved zip file. Default “dataset.zip”.
        chunk_size (int, optional): The chunk size for the download. Default 1024 (1 KB).
        is_file_removed (bool, optional): Delete the zip file after extraction. Default True.
    Raises:
        Exception: If there is an error while downloading or extracting the file.
    """

    # Memastikan direktori penyimpanan dan ekstrak ada
    os.makedirs(save_dir_path, exist_ok=True)
    os.makedirs(extract_dir_path, exist_ok=True)

    # Path file zip yang akan disimpan
    zipfile_path = os.path.join(save_dir_path, filename)

    # Jika belum ada hasil ekstrak, lanjutkan unduhan
    try:
        if not os.path.exists(zipfile_path):
            # Mengunduh file dengan progress bar
            response = requests.get(url, stream=True, timeout=10)
            response.raise_for_status()  # Cek jika ada error HTTP

            total_size = int(response.headers.get('content-length', 0))
            with open(zipfile_path, "wb") as file, tqdm(
                    desc=f"Downloading {filename}",
                    total=total_size,
                    unit='B', unit_scale=True, unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    file.write(chunk)
                    bar.update(len(chunk))

            display_log_message(f"File downloaded to {color_text(zipfile_path)}", is_log=False)
        else:
            display_log_message(f"{filename} already in {color_text(zipfile_path)}.", is_log=False)

        # Mengekstrak file zip jika belum ada hasil ekstrak
        with zipfile.ZipFile(zipfile_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir_path)
            display_log_message(f"Files extracted to {color_text(extract_dir_path)}", is_log=False)

        # Menghapus file zip yang sudah diekstrak jika is_file_removed=True
        if os.path.exists(zipfile_path) and is_file_removed:
            os.remove(zipfile_path)
            display_log_message("Downloaded zip file removed.", is_log=False)

        return

    except requests.exceptions.RequestException as e:
        raise Exception(f"Error downloading the file: {e}")

    except zipfile.BadZipFile:
        raise Exception("Error: The downloaded file is not a valid zip file.")

    except Exception as e:
        raise TrashClassificationException(e, sys)