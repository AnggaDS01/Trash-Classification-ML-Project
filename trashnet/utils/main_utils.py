import os 
import inspect 
import re 
import dill 
import sys
import yaml
import numpy as np

from pathlib import Path
from box import ConfigBox
from box.exceptions import BoxValueError
from colorama import Fore, Style
from trashnet.logger import logging
from trashnet.exception import TrashClassificationException

def color_text(
        text: str = None, 
        color: str = Fore.YELLOW, 
        reset: bool = True
    ) -> str:
    """
    Color the given text with the specified ANSI color code.

    Args:
        text (str): The text to color.
        color (str): ANSI color code to apply. Defaults to yellow.
        reset (bool): Flag to reset color after the text. Defaults to True.

    Returns:
        str: The colored text.
    
    Raises:
        TrashClassificationException: If an error occurs during text coloring.
    """
    try:
        # Construct colored text with or without reset based on the flag
        colored_text = f"{color}{text}{Style.RESET_ALL}" if reset else f"{color}{text}"
        return colored_text

    except Exception as e:
        # Raise custom exception with traceback details
        raise TrashClassificationException(e, sys)

def remove_color_codes(text: str = None) -> str:
    """
    Remove ANSI color codes from the given text.

    Args:
        text (str): The text from which ANSI color codes should be removed.

    Returns:
        str: The text without ANSI color codes.
    """
    try:
        # Regular expression to match ANSI escape sequences
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        
        # Remove ANSI escape sequences from the text
        return ansi_escape.sub('', text)
    
    except Exception as e:
        raise TrashClassificationException(e, sys)

def show_data_info(**datasets) -> None:
    """
    Function to display information about the given datasets.

    Args:
        **datasets (dict): A dictionary of datasets to be displayed.

    Raises:
        TrashClassificationException: If an exception occurs while displaying the dataset information.
    """
    try:
        # Iterate over the given datasets
        for dataset_name, dataset in datasets.items():
            # Print the title of the dataset
            custom_title_print(f"{dataset_name} info")
            # Print the information of the dataset
            print(f'info {dataset_name}: {dataset}')
            # Print the number of elements in the dataset
            print(f'number of {dataset_name}: {len(dataset)}')
            # Print a newline for better readability
            print()

    except Exception as e:
        raise TrashClassificationException(e, sys)

def display_log_message(msg, is_log=True) -> None:
    """
    Function to print a message to the console and log it to the log file.

    Args:
        msg (str): The message to be printed and logged.
        is_log (bool): A flag indicating whether the message should be logged or not.

    Raises:
        TrashClassificationException: If an exception occurs while displaying the message.
    """
    try:
        # Print the message to the console
        print(f"{color_text('[INFO]', color=Fore.GREEN)} {msg}")
        # If is_log is True, log the message to the log file
        if is_log:
            # Remove ANSI escape sequences from the message
            msg_no_color = remove_color_codes(msg)
            # Log the message
            logging.info(msg_no_color)

    except Exception as e:
        # Raise an exception if any error occurs
        raise TrashClassificationException(e, sys)

def display_function_info(frame) -> tuple:
    """
    Retrieves the name of the function, class, and file location 
    of the calling function for display purposes.

    Args:
        frame (frame): A frame object representing the function 
                       or method to display information for.

    Returns:
        tuple: A tuple containing the function name, class name 
               (or None if not in a class), and file name.
    """
    try:
        # Retrieve the function's name and the file in which it is defined
        function_name = frame.f_code.co_name
        file_name = inspect.getfile(frame)
        
        # Check if there is a class context and retrieve class name if exists
        class_name = frame.f_locals.get('self')
        class_name = class_name.__class__.__name__ if class_name else None
        
        return function_name, class_name, file_name

    except Exception as e:
        raise TrashClassificationException(e, sys)

def custom_title_print(
        title: str='Title', 
        n_strip: int=110
    ) -> None:

    """
    Mencetak judul yang disesuaikan dengan garis pembatas di atas dan di bawah judul.

    Args:
        title (str): Judul yang ingin ditampilkan.
        n_strip (int): Jumlah karakter '=' untuk membuat garis pembatas. Default adalah 80.

    Returns:
        None
    """

    try:
        title = f'''{'=' * n_strip}
    {title.upper().center(n_strip, '=')}
    {'=' * n_strip}'''

        print(title)

    except Exception as e:
        raise TrashClassificationException(e, sys)

def save_object(
        file_path: str=None, 
        obj: object=None
    ) -> None:

    """
    Menyimpan objek ke dalam file menggunakan serialisasi dengan dill.

    Args:
        file_path (str): Jalur file tempat objek akan disimpan.
        obj (object): Objek yang akan disimpan, bisa berupa list, dictionary, atau objek Python lainnya.

    Returns:
        None
    """

    try:
        if os.path.exists(file_path): # Memeriksa apakah file sudah ada
            display_log_message(f"File '{color_text(file_path)}' already exists. Skipping saving.")
            return # Jika sudah ada, tidak perlu menyimpan lagi

        with open(file_path, 'wb') as file_obj: # Membuka file dalam mode write-binary
            dill.dump(obj, file_obj) # Menyimpan objek menggunakan dill
        print(f"Object saved to {file_path}")

    except Exception as e:
        raise TrashClassificationException(e, sys)
    
def load_object(file_path: str=None) -> object:
    """
    Fungsi untuk memuat objek dari file dengan menggunakan modul `dill`.

    Args:
        file_path (str): Lokasi file dari objek yang ingin dimuat.

    Returns:
        object: Objek yang dimuat dari file.
    """

    try:
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        raise TrashClassificationException(e, sys)

def read_yaml(path_to_yaml: Path=None) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise TrashClassificationException(e, sys)

def create_directories(
        path_to_directories: list=[], 
        verbose: bool=True
    ) -> None:

    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    try:
        for path in path_to_directories:
            os.makedirs(path, exist_ok=True)
            if verbose:
                display_log_message(f"created directory at: {color_text(path)}")

    except Exception as e:
        raise TrashClassificationException(e, sys)


class DataInspector:
    """
    Kelas `DataInspector` bertanggung jawab untuk melakukan inspeksi dan visualisasi
    gambar dalam dataset pelatihan, validasi, dan pengujian.
    """

    def __init__(self, label_encoding):
        """
        Inisialisasi kelas `DataInspector`.

        Args:
            label_encoding (dict): Mapping dari label numerik ke label kelas.
        """
        self.label_encoding = label_encoding

    def _inspect_single_dataset(self, dataset, ds_name, ispath, idx):
        """
        Helper function untuk menginspeksi dataset tertentu.

        Args:
            dataset (tf.data.Dataset): Dataset yang akan diinspeksi.
            ds_name (str): Nama dataset (train, valid, test).
            idx (int, optional): Indeks untuk memulai pengambilan contoh gambar. Default 1.
        """

        if ispath:
            for i, (image, label, path) in enumerate(dataset.skip(idx).take(1), 1):
                self._print_data_info(f"{ds_name}_data info", image, label, path)
        else:
            for i, (image, label) in enumerate(dataset.skip(idx).take(1), 1):
                self._print_data_info(f"{ds_name}_data info", image, label)

    def _print_data_info(self, title, image, label, image_path=None):
        """
        Menampilkan informasi mendetail tentang gambar dan label.

        Args:
            title (str): Judul informasi yang akan ditampilkan.
            image (tf.Tensor): Gambar yang diinspeksi.
            label (int): Label gambar yang diinspeksi.
            image_path (str, optional): Jalur file gambar (jika ada). Default adalah None.
        """
        print('\n\n')
        custom_title_print(title)

        if image_path is not None:
            print(f'image path: {image_path}')

        print(f'shape-image: {image.shape}')
        print(f'dtype-image: {image.dtype}')
        print(f'max-intensity: {np.max(image)}')
        print(f'min-intensity: {np.min(image)}')

        print(f'label: {label} -> {self.label_encoding[label.numpy()]}')
        print(f'label-shape: {label.shape}')
        print(f'label-type: {label.dtype}')
        print()

    def inspect(self, ispath=False, idx=1, **datasets):
        """
        Menginspeksi gambar dari dataset pelatihan, validasi, atau pengujian (atau gabungan).

        Args:
            datasets (dict): Dataset yang ingin diinspeksi (train_ds, valid_ds, test_ds).
                             Bisa masukkan satu atau lebih.
        """
        # Looping dinamis sesuai dataset yang diberikan (train, valid, test)
        for ds_name, ds in datasets.items():
            self._inspect_single_dataset(ds, ds_name, ispath, idx)

