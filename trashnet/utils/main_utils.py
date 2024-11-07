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
from colorama import init, Fore, Back, Style
from colorama import Fore, Style
from trashnet.logger import logging
from trashnet.exception import TrashClassificationException

def color_text(text, color=Fore.YELLOW, reset=True):
    """
    Function to change the text color according to the color parameter.
    
    Parameters:
    text (str): The text to be colored.
    color (str): The color to apply, default is yellow.
    reset (bool): Sets whether the color will be reset back to default after the text, default is True.
    Returns:
    str: The text that has been colored.
    """

    colored_text = f"{color}{text}{Style.RESET_ALL}" if reset else f"{color}{text}"
    return colored_text

def remove_color_codes(text):
    """
    Remove color codes (ANSI escape codes) from text.
    
    Parameters:
    text (str): Text that may contain color codes.
    
    Returns:
    str: Text without color coding.
    """

    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    return ansi_escape.sub('', text)

def display_log_message(msg, is_log=True):
    print(f"{color_text('[INFO]', color=Fore.GREEN)} {msg}")
    if is_log:
        msg_no_color = remove_color_codes(msg)
        logging.info(msg_no_color)

def display_function_name(function_) -> None:
    """
    Displays the name and file location of a given function.

    Args:
        function_ (frame): A frame object representing the function to display information for.

    Returns:
        None
    """
    
    # Retrieve the function's name and the file in which it is defined
    function_name = function_.f_code.co_name
    file_name_function = inspect.getfile(function_)
    
    return function_name, file_name_function

def custom_title_print(title, n_strip=110):
    """
    Mencetak judul yang disesuaikan dengan garis pembatas di atas dan di bawah judul.

    Args:
        title (str): Judul yang ingin ditampilkan.
        n_strip (int): Jumlah karakter '=' untuk membuat garis pembatas. Default adalah 80.

    Returns:
        None
    """

    title = f'''{'=' * n_strip}
{title.upper().center(n_strip, '=')}
{'=' * n_strip}'''

    print(title)

def save_object(file_path, obj):
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

        dir_path = os.path.dirname(file_path)  # Mendapatkan jalur direktori dari file
        os.makedirs(dir_path, exist_ok=True) # Membuat direktori jika belum ada

        with open(file_path, 'wb') as file_obj: # Membuka file dalam mode write-binary
            dill.dump(obj, file_obj) # Menyimpan objek menggunakan dill
        print(f"Object saved to {file_path}")

    except Exception as e:
        raise TrashClassificationException(e, sys)
    
def load_object(file_path):
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

def read_yaml(path_to_yaml: Path) -> ConfigBox:
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
        raise e

# Fungsi untuk mengecek keberadaan file
def paths_exist(paths):
    missing_paths = [path for path in paths if not os.path.exists(path)]
    
    if missing_paths:
        for path in missing_paths:
            logging.info(f"{path} not found.")
        return False
    return True

class DataInspector:
    """
    Kelas `DataInspector` bertanggung jawab untuk melakukan inspeksi dan visualisasi
    gambar dalam dataset pelatihan, validasi, dan pengujian.
    """

    def __init__(self, label_encoding, figsize):
        """
        Inisialisasi kelas `DataInspector`.

        Args:
            label_encoding (dict): Mapping dari label numerik ke label kelas.
            figsize (tuple): Ukuran figure untuk plot visualisasi gambar.
        """
        self.label_encoding = label_encoding
        self.figsize = figsize

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

