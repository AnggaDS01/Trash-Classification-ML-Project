import os
import tensorflow as tf

from PIL import Image
from trashnet.utils.main_utils import custom_title_print


class FilePathInfo:
    """
    The FilePathInfo class is used to display detailed information about file paths on a dataset, including
    file name, extension, file size, and label (if any). This class also supports the use of different
    size units such as 'bytes', 'kb', 'mb', and 'gb'.
    Args:
        unit_file_size (str, optional): The unit to display the file size ('bytes', 'kb', 'mb', 'gb'). Default is 'bytes'.
    """

    def __init__(
            self, 
            unit_file_size='bytes'
        ):

        """
        Initialize the FilePathInfo class with the given file size unit.
        Args:
            unit_file_size (str, optional): The unit to display the file size ('bytes', 'kb', 'mb', 'gb'). Default is 'bytes'.
        """
        self.unit_file_size = unit_file_size.lower()
        self.units = ['bytes', 'kb', 'mb', 'gb']
        if self.unit_file_size not in self.units:
            raise ValueError(f"Invalid unit. Choose from {self.units}.")

    def show_train_files_path_info(
            self, 
            files_path_data, 
            is_labeled=True, 
            is_random=False
        ):

        """
        Display detailed information about file paths in the training dataset.
        Args:
            files_path_data(tf.data.Dataset): The dataset containing the file paths.
            is_labeled (bool, optional): Indicates whether the dataset has a label. Default is True.
            is_random (bool, optional): Indicates whether the dataset needs to be randomized before displaying. Default is False.
        Returns:
            int: The label index on the file path, if the dataset has a label.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        label_index = self._display_path_info(files_path_data_plot, is_labeled)
        return label_index

    def show_test_files_path_info(
            self, 
            files_path_data, 
            is_labeled=False, 
            is_random=False
        ):

        """
        Display detailed information about file paths in the testing dataset.
        Args:
            files_path_data(tf.data.Dataset): The dataset containing the file paths.
            is_random (bool, optional): Indicates whether the dataset needs to be randomized before display. Default is False.
        """
        files_path_data_plot = self._get_files_path_data_plot(files_path_data, is_random)
        self._display_path_info(files_path_data_plot, is_labeled)

    def _get_files_path_data_plot(
            self, 
            files_path_data, 
            is_random
        ):

        """
        Retrieves a subset of the file path dataset, with randomization options.
        Args:
            files_path_data(tf.data.Dataset): The dataset containing the file paths.
            is_random (bool): Whether the dataset needs to be randomized.
        Returns:
            tf.data.Dataset: A subset of the selected dataset.
        """
        if is_random:
            return files_path_data.shuffle(buffer_size=files_path_data.cardinality().numpy()).take(1)
        else:
            return files_path_data.take(1)

    def _display_path_info(
            self, 
            files_path_data_plot, 
            is_labeled
        ):

        """
        Displays the full information of the selected file path, including the file name, extension, size, and label if any.
        Args:
            files_path_data_plot(tf.data.Dataset): A subset of the dataset to be displayed.
            is_labeled (bool): Indicates whether the dataset has a label.
        Returns:
            int: The label index on the file path if the dataset is labeled.
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

    def _split_file_path(
            self, 
            file_path
        ):

        """
        Breaks the file path into parts using the file system separator.
        Args:
            file_path (tf.Tensor): File path.
        Returns:
            tf.Tensor: The split file path.
        """
        return tf.strings.split(file_path, os.path.sep)

    def _display_split_file_path(
            self, 
            split_file_path
        ):

        """
        Displays the path of the split file and the index of each section.
        Args:
            split_file_path(tf.Tensor): The path of the split file.
        """
        custom_title_print(' SPLIT FILE PATH ')
        print(f'Split File Path: {split_file_path}')
        print()

        custom_title_print(' INDEXED PATH ')
        result = {value: f'Index -> {index}' for index, value in enumerate(split_file_path.numpy())}
        for key, value in result.items():
            print(f'{value}: {key}')
        print()

    def _display_kind_data_info(
            self, 
            split_file_path, 
            kind_data
        ):

        """
        Displays the index and label of the data based on its type.
        Args:
            split_file_path(tf.Tensor): The path of the split file.
            kind_data (str): The type of data in the file path.
        Returns:
            int: The index of the label in the file path.
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

    def _display_file_info(
            self, 
            split_file_path, 
            file_path
        ):

        """
        Displays detailed information about the file such as name, extension, and file size.
        Args:
            split_file_path(tf.Tensor): The path of the split file.
            file_path (tf.Tensor): The path of the file.
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


    def _format_file_size(
            self, 
            size
        ):
        
        """
        Formats the file size according to the selected unit.
        Args:
            size (int): File size in bytes.
        Returns:
            str: The size of the formatted file.
        """
        if self.unit_file_size == 'kb':
            size /= 1024
        elif self.unit_file_size == 'mb':
            size /= 1024 ** 2
        elif self.unit_file_size == 'gb':
            size /= 1024 ** 3

        return f'{size:.4f}' if self.unit_file_size != 'bytes' else size