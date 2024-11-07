import sys
import re 
import random 
import os
import tensorflow as tf
import numpy as np

from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
from PIL import Image
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from trashnet.exception import TrashClassificationException

from trashnet.utils.main_utils import (color_text,
                                       custom_title_print)

def print_class_distribution(
        distribution
    ):

    """
    Mencetak distribusi kelas ke layar.

    Args:
        distribution (collections.Counter): Objek Counter yang berisi distribusi kelas.
    """
    for class_name, count in sorted(distribution.items()):
        print(f"{class_name}: {count}")

def extract_class_from_path_tf(
        path_tensor
    ):

    """
    Mengekstrak nama kelas dari jalur file yang disimpan dalam bentuk tensor.

    Args:
        path_tensor (tf.Tensor): Tensor yang berisi jalur file.

    Returns:
        tf.Tensor: Tensor yang berisi nama kelas yang diekstrak dari jalur file.
    """

    try:
        parts = tf.strings.split(path_tensor, os.path.sep)
        class_name = parts[-2]  # Nama kelas biasanya berada di posisi kedua dari belakang dalam jalur
        return class_name
    except Exception as e:
        print(f"Error: {e}")

def calculate_class_distribution_tf(
        dataset, 
        class_labels):
    """
    Menghitung distribusi kelas dan class weight menggunakan `compute_class_weight` dari sklearn.

    Args:
        dataset (tf.data.Dataset): Dataset TensorFlow berisi gambar dan label.
        class_labels (list): Daftar nama kelas yang diurutkan (misal: ['cardboard', 'glass', ...]).

    Returns:
        tuple: class_counts (Counter), class_weights (dict)
            class_weights dalam bentuk {label_index: weight}
    """
    try:
        # Mengambil nama kelas dari path dan mengonversinya menjadi list
        class_names = dataset.map(lambda x: extract_class_from_path_tf(x))
        class_names_list = list(class_names.batch(1000).as_numpy_iterator())
        all_class_names = [name.decode('utf-8') for batch in class_names_list for name in batch]

        # Hitung distribusi kelas
        class_counts = Counter(all_class_names)

        # Konversi nama kelas ke indeks numerik sesuai `class_labels`
        class_indices = [class_labels.index(name) for name in all_class_names]

        # Hitung class weight menggunakan sklearn
        class_weight_values = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(class_indices),
            y=class_indices
        )

        # Buat dict class_weights sesuai indeks
        class_weights = {i: weight for i, weight in enumerate(class_weight_values)}

        return class_counts, class_weights

    except Exception as e:
        print(f"Error: {e}")

def collect_images_with_regex_and_count(
        path, 
        folders, 
        extensions_pattern
    ):

    """
    Mengumpulkan jalur gambar dari beberapa folder yang ada dalam path utama dengan memfilter gambar
    berdasarkan ekstensi menggunakan regex. Fungsi ini mengembalikan dictionary dengan kunci berupa
    nama folder dan nilai berupa daftar jalur file gambar.

    Args:
        path (str): Jalur utama folder yang berisi sub-folder data gambar.
        folders (list): Daftar nama folder yang akan di-scan untuk gambar.
        extensions_pattern (str): Pola regex untuk mencocokkan ekstensi file gambar (contoh: r'\.(jpg|png|jpeg)$').

    Returns:
        dict: Dictionary dengan kunci berupa nama folder dan nilai berupa daftar jalur file gambar yang sesuai dengan pola.
    """

    try:
        image_paths = {folder: [] for folder in folders}   # Membuat dictionary kosong untuk menyimpan jalur gambar
        pattern = re.compile(extensions_pattern, re.IGNORECASE) # Membuat pola regex untuk mencocokkan ekstensi gambar dengan ignore case

        for folder in folders:
            folder_path = Path(path) / folder # Mendapatkan jalur folder

            for file_path in tqdm(folder_path.rglob("*"), desc=f"Collecting from {folder}", unit=" paths"):
                if pattern.search(file_path.suffix):  # Memeriksa apakah ekstensi file cocok dengan pola
                    image_paths[folder].append(file_path)  # Menambahkan jalur file ke dalam daftar jika cocok

        return image_paths  # Mengembalikan dictionary jalur gambar

    except Exception as e:
        print(f"No classes are retrieved from directory validation")
        return {}
    
def get_random_images(
        image_paths, 
        num_samples, 
        seed=42
    ):
    """
    Mengambil sejumlah gambar secara acak dari daftar jalur gambar.

    Args:
        image_paths (list): Daftar jalur gambar.
        num_samples (int): Jumlah gambar yang ingin diambil. Jika None, semua gambar akan dipilih.
        seed (int): Seed untuk mengontrol hasil pengambilan acak agar hasilnya bisa direproduksi. Default adalah 42.

    Returns:
        list: Daftar jalur gambar yang dipilih secara acak.
    """
    random.seed(seed)
    return random.sample(image_paths, min(len(image_paths) if num_samples is None else num_samples, len(image_paths)))


def collect_and_combine_images(
        classes, 
        train_path=None, 
        valid_path=None, 
        pattern_regex=r"\.(jpe?g)$", 
        num_images_per_class=None, 
        seed=42
    ):
    """
    Mengumpulkan dan menggabungkan gambar dari folder training dan validation, lalu mengambil sejumlah gambar secara acak dari setiap kelas.

    Args:
        classes (list): Daftar kelas (nama folder) yang ingin diproses.
        train_path (str): Jalur utama folder training yang berisi sub-folder data gambar.
        valid_path (str): Jalur utama folder validation yang berisi sub-folder data gambar.
        pattern_regex (str): Pola regex untuk mencocokkan ekstensi file gambar (contoh: r'\.(jpg|png|jpeg)$').
        num_images_per_class (dict): Dictionary berisi jumlah gambar yang ingin diambil untuk setiap kelas. Jika None, semua gambar akan diambil.
        seed (int): Seed untuk pengambilan gambar secara acak. Default adalah 42.

    Returns:
        list: Daftar gabungan jalur gambar dari folder training dan validation yang diambil secara acak.
    """

    try:
        def process_class(cls):
            # Menggabungkan gambar dari training dan validation untuk setiap kelas
            all_train_images = train_images_paths.get(cls, [])
            all_valid_images = valid_images_paths.get(cls, [])
            all_combined_images = all_train_images + all_valid_images

            # Mengambil sejumlah gambar acak dari gambar gabungan
            return get_random_images(
                image_paths=all_combined_images,
                num_samples=None if num_images_per_class is None else num_images_per_class.get(cls, len(all_combined_images)),
                seed=seed
            )

        custom_title_print(f"COLLECT {classes} FROM TRAINING DATA")
        train_images_paths = collect_images_with_regex_and_count(train_path, classes, pattern_regex)
        custom_title_print(f"=")
        print()

        # Mencetak judul untuk proses pengumpulan gambar dari data validation
        custom_title_print(f"COLLECT {classes} FROM VALIDATION DATA")
        valid_images_paths = collect_images_with_regex_and_count(valid_path, classes, pattern_regex)
        custom_title_print(f"=")
        print()

        # Mencetak judul untuk proses penggabungan gambar dari training dan validation
        custom_title_print(f"COMBINING {classes} FROM TRAINING AND VALIDATION DATA")

        random_images = {}

        # Menggunakan ThreadPoolExecutor untuk mempercepat proses pengambilan gambar dari setiap kelas secara paralel
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_class, classes)

        # Menyimpan hasil gambar acak untuk setiap kelas ke dalam dictionary
        for cls, images in zip(classes, results):
            random_images[cls] = images
            print(f"Total {cls} taken: {color_text(len(random_images[cls]))}")

        # Menggabungkan semua jalur gambar dari semua kelas
        all_images_paths = sum(random_images.values(), [])
        all_images_paths = [str(path) for path in all_images_paths]
        custom_title_print(f"Total images taken: {len(all_images_paths)}")

        return all_images_paths

    except Exception as e:
        raise TrashClassificationException(e, sys)
    
def create_label_list_table(
        label_list, 
        default_value=-1
    ):
    """
    Membuat StaticHashTable untuk encoding label berdasarkan daftar label yang diberikan.

    Args:
    - label_list (list): Daftar label yang ingin di-encode.
    - default_value (int): Nilai default jika label tidak ditemukan di dalam tabel.

    Returns:
    - label_table (tf.lookup.StaticHashTable): Tabel hash untuk label encoding.
    """
    # Buat tensor dari daftar label (keys)
    keys_tensor = tf.constant(label_list)

    # Buat nilai integer yang berkaitan dengan setiap label (values)
    values_tensor = tf.range(len(label_list))

    # Inisialisasi Key-Value pairs untuk tabel hash
    table_init = tf.lookup.KeyValueTensorInitializer(keys_tensor, values_tensor)

    # Membuat StaticHashTable
    label_table = tf.lookup.StaticHashTable(table_init, default_value=default_value)

    return label_table



class FilePathInfo:
    """
    Kelas FilePathInfo digunakan untuk menampilkan informasi detail tentang jalur file pada dataset, termasuk
    nama file, ekstensi, ukuran file, dan label (jika ada). Kelas ini juga mendukung penggunaan unit ukuran
    file yang berbeda seperti 'bytes', 'kb', 'mb', dan 'gb'.

    Args:
        unit_file_size (str, optional): Unit untuk menampilkan ukuran file ('bytes', 'kb', 'mb', 'gb'). Default adalah 'bytes'.
    """

    def __init__(self, unit_file_size='bytes'):
        """
        Inisialisasi kelas FilePathInfo dengan unit ukuran file yang diberikan.

        Args:
            unit_file_size (str, optional): Unit untuk menampilkan ukuran file ('bytes', 'kb', 'mb', 'gb'). Default adalah 'bytes'.
        """
        self.unit_file_size = unit_file_size.lower()
        self.units = ['bytes', 'kb', 'mb', 'gb']
        if self.unit_file_size not in self.units:
            raise ValueError(f"Invalid unit. Choose from {self.units}.")

    def show_train_files_path_info(self, files_path_data, is_labeled=True, is_random=False):
        """
        Menampilkan informasi detail tentang jalur file pada dataset training.

        Args:
            files_path_data (tf.data.Dataset): Dataset yang berisi jalur file.
            is_labeled (bool, optional): Menunjukkan apakah dataset memiliki label. Default adalah True.
            is_random (bool, optional): Menunjukkan apakah dataset perlu diacak sebelum ditampilkan. Default adalah False.

        Returns:
            int: Indeks label pada jalur file, jika dataset memiliki label.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        label_index = self._display_path_info(files_path_data_plot, is_labeled)
        return label_index

    def show_test_files_path_info(self, files_path_data, is_labeled=False, is_random=False):
        """
        Menampilkan informasi detail tentang jalur file pada dataset testing.

        Args:
            files_path_data (tf.data.Dataset): Dataset yang berisi jalur file.
            is_random (bool, optional): Menunjukkan apakah dataset perlu diacak sebelum ditampilkan. Default adalah False.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        self._display_path_info(files_path_data_plot, is_labeled)

    def _get_files_path_data_plot(self, files_path_data, is_random):
        """
        Mengambil subset dari dataset jalur file, dengan opsi pengacakan.

        Args:
            files_path_data (tf.data.Dataset): Dataset yang berisi jalur file.
            is_random (bool): Apakah dataset perlu diacak.

        Returns:
            tf.data.Dataset: Subset dari dataset yang dipilih.
        """
        if is_random:
            return files_path_data.shuffle(buffer_size=files_path_data.cardinality().numpy()).take(1)
        else:
            return files_path_data.take(1)

    def _display_path_info(self, files_path_data_plot, is_labeled):
        """
        Menampilkan informasi lengkap dari jalur file yang dipilih, termasuk nama file, ekstensi, ukuran, dan label jika ada.

        Args:
            files_path_data_plot (tf.data.Dataset): Subset dari dataset yang akan ditampilkan.
            is_labeled (bool): Menunjukkan apakah dataset memiliki label.

        Returns:
            int: Indeks label pada jalur file jika dataset berlabel.
        """
        for file_path in files_path_data_plot:
            custom_title_print(' PATH INFO ')
            print(f'File Path: {file_path.numpy().decode("utf-8")}')
            print()

            split_file_path = self._split_file_path(file_path)
            self._display_split_file_path(split_file_path)

            if is_labeled:
                kind_data = split_file_path[-3].numpy().decode('utf-8')
                index_label = self._display_kind_data_info(split_file_path, kind_data)
                self._display_file_info(split_file_path, file_path)
                return index_label
            else:
                self._display_file_info(split_file_path, file_path)

    def _split_file_path(self, file_path):
        """
        Memecah jalur file menjadi bagian-bagian menggunakan separator file system.

        Args:
            file_path (tf.Tensor): Jalur file.

        Returns:
            tf.Tensor: Jalur file yang sudah dipecah.
        """
        return tf.strings.split(file_path, os.path.sep)

    def _display_split_file_path(self, split_file_path):
        """
        Menampilkan jalur file yang sudah dipecah dan indeks dari setiap bagian.

        Args:
            split_file_path (tf.Tensor): Jalur file yang sudah dipecah.
        """
        custom_title_print(' SPLIT FILE PATH ')
        print(f'Split File Path: {split_file_path}')
        print()

        custom_title_print(' INDEXED PATH ')
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path.numpy())}
        for key, value in result.items():
            print(f'{value}: {key}')
        print()

    def _display_kind_data_info(self, split_file_path, kind_data):
        """
        Menampilkan indeks dan label dari data berdasarkan jenisnya.

        Args:
            split_file_path (tf.Tensor): Jalur file yang sudah dipecah.
            kind_data (str): Jenis data yang ada di jalur file.

        Returns:
            int: Indeks label pada jalur file.
        """
        custom_title_print(f' KIND DATA INDEX {kind_data} ')
        index = tf.where(tf.equal(split_file_path, kind_data))[0][0]
        print(f'Index of "{kind_data}": {index}')
        print()

        index_label = index + 1
        custom_title_print(' INDEX LABEL ')
        print(f'Index Label: {index_label}')
        print()

        custom_title_print(' LABEL ')
        print(f'Label: {split_file_path[index_label]}')
        print()

        return index_label.numpy()

    def _display_file_info(self, split_file_path, file_path):
        """
        Menampilkan informasi detail tentang file seperti nama, ekstensi, dan ukuran file.

        Args:
            split_file_path (tf.Tensor): Jalur file yang sudah dipecah.
            file_path (tf.Tensor): Jalur file.
        """
        file_name = split_file_path[-1].numpy().decode('utf-8')
        custom_title_print(' FILE NAME ')
        print(f'File Name: {file_name}')
        print()

        file_extension = os.path.splitext(file_name)[1]
        custom_title_print(' FILE EXTENSION ')
        print(f'File Extension: {file_extension}')
        print()

        image_size = Image.open(file_path.numpy().decode('utf-8')).size
        custom_title_print(' IMAGE SIZE (PX)')
        print(f'Image Size: \n width={image_size[0]} \n height={image_size[1]}')
        print()

        file_size = os.path.getsize(file_path.numpy().decode('utf-8'))
        file_size = self._format_file_size(file_size)
        custom_title_print(' FILE SIZE ')
        print(f'File Size: {file_size} {self.unit_file_size}')
        print()


    def _format_file_size(self, size):
        """
        Memformat ukuran file sesuai dengan unit yang dipilih.

        Args:
            size (int): Ukuran file dalam bytes.

        Returns:
            str: Ukuran file yang sudah diformat.
        """
        if self.unit_file_size == 'kb':
            size /= 1024
        elif self.unit_file_size == 'mb':
            size /= 1024 ** 2
        elif self.unit_file_size == 'gb':
            size /= 1024 ** 3

        return f'{size:.4f}' if self.unit_file_size != 'bytes' else size
    
class DatasetSplitter:
    """
    Kelas ini digunakan untuk membagi dataset menjadi tiga bagian: training, validation, dan testing.
    Pembagian dilakukan berdasarkan rasio yang dapat dikonfigurasi, dan dataset dapat diacak sebelum dibagi.
    """

    def __init__(self):
        """
        Inisialisasi kelas `DatasetSplitter`. Tidak ada argumen yang diterima saat inisialisasi.
        """
        pass

    def split_train_valid_test(self, dataset, split_ratio=None, shuffle=True, buffer_size=None, seed=42):
        """
        Membagi dataset menjadi tiga bagian: training, validation, dan testing.

        Args:
            dataset (tf.data.Dataset): Dataset yang akan dibagi.
            train_ratio (float, optional): Rasio data untuk training. Default adalah 0.7.
            valid_ratio (float, optional): Rasio data untuk validation. Default adalah 0.2.
            shuffle (bool, optional): Apakah dataset perlu diacak sebelum pembagian. Default adalah True.
            buffer_size (int, optional): Ukuran buffer untuk pengacakan dataset. Jika None, buffer_size diambil dari ukuran dataset.
            seed (int, optional): Seed untuk pengacakan dataset. Default adalah 42.

        Returns:
            tuple: Tuple yang berisi tiga dataset yang sudah dibagi: (train_dataset, val_dataset, test_dataset).
        """

        try:
            dataset_size = len(dataset) if buffer_size is None else buffer_size

            # Define individual ratios for training, validation, and test sets
            train_ratio = split_ratio[0]
            val_ratio = split_ratio[1]
            test_ratio = round(max(1.0 - (train_ratio + val_ratio), 0), 4)  # Calculate remaining ratio for test set

            # Verify the total ratio equals 1.0; raise an error if not
            total_ratio = round(sum((train_ratio, val_ratio, test_ratio)), 2)
            if total_ratio != 1.0:
                raise ValueError("[ERROR] split_ratio must sum to 1.0.\n")

            # Determine the number of images in each split based on the calculated ratios
            train_size = int(round(dataset_size * train_ratio, 0))
            val_size = int(round(dataset_size * val_ratio, 0))
            test_size = int(round(dataset_size * test_ratio, 0))

            # Randomly shuffle the image files if random_split is enabled
            if shuffle:
                dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed)

            # Split the files into training, validation, and test sets based on calculated sizes
            train_dataset = dataset.take(train_size)
            val_test_dataset = dataset.skip(train_size)

            # Jika test_size == 0, hanya buat train dan validation dataset
            if test_size == 0:
                val_dataset = val_test_dataset.take(val_size)
                self._display_info(
                    dataset=dataset,
                    train_dataset=train_dataset,
                    valid_dataset=val_dataset,
                    dataset_size=dataset_size,
                    shuffle=shuffle,
                    test_size=test_size
                )

                return train_dataset, val_dataset
            else:
                val_dataset = val_test_dataset.take(val_size)
                test_dataset = val_test_dataset.skip(val_size)

                self._display_info(
                    dataset=dataset,
                    train_dataset=train_dataset,
                    valid_dataset=val_dataset,
                    test_dataset=test_dataset,
                    dataset_size=dataset_size,
                    shuffle=shuffle,
                    test_size=test_size
                )

                return train_dataset, val_dataset, test_dataset

        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def _display_info(self, dataset=None, train_dataset=None, valid_dataset=None, test_dataset=None, dataset_size=None, shuffle=False, test_size=None):
        """
        Menampilkan informasi mengenai pembagian dataset seperti ukuran, rasio, dan status shuffle.

        Args:
            dataset (tf.data.Dataset): Dataset asli yang belum dibagi.
            train_dataset (tf.data.Dataset): Dataset bagian training.
            valid_dataset (tf.data.Dataset): Dataset bagian validation.
            test_dataset (tf.data.Dataset): Dataset bagian testing.
            dataset_size (int): Ukuran total dataset.
            shuffle (bool): Status apakah dataset diacak sebelum dibagi.
        """
        train_ratio = len(train_dataset) / dataset_size
        valid_ratio = len(valid_dataset) / dataset_size
        test_ratio = len(test_dataset) / dataset_size if test_size > 0 else 0

        print(f"Total number of data: {dataset_size}")
        print(f"Shuffle status: {shuffle}")

        custom_title_print(' Training Dataset ')
        print(f"Info data: {train_dataset}")
        print(f"Training Split: {round(train_ratio * 100, 2)}%")
        print(f"Number of data: {len(train_dataset)}")
        print()

        custom_title_print(' Validation Dataset ')
        print(f"Info data: {valid_dataset}")
        print(f"Validation Split: {round(valid_ratio * 100, 2)}%")
        print(f"Number of data: {len(valid_dataset)}")
        print()

        if test_size > 0:
            custom_title_print(' Test Dataset ')
            print(f"Info data: {test_dataset}")
            print(f"Test Split: {round(test_ratio * 100, 2)}%")
            print(f"Number of data: {len(test_dataset)}")