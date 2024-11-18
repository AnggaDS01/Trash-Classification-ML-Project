import sys
import inspect 
import tensorflow as tf
import numpy as np
import random

from trashnet.components.data_preprocessing_components.utils.split_data import DatasetSplitter
from trashnet.components.data_preprocessing_components.utils.display_filepath_info import FilePathInfo
from trashnet.components.data_preprocessing_components.utils.image_processor import ImagePreprocessor
from trashnet.exception import TrashClassificationException
from trashnet.components.data_preprocessing_components.utils.collect_images_paths import collect_and_combine_images
from trashnet.components.data_preprocessing_components.utils.calculate_distribution_class import calculate_class_distribution_tf
from trashnet.components.data_preprocessing_components.utils.display_distribution_class import print_class_distribution
from trashnet.components.data_preprocessing_components.utils.create_label_list_table import create_label_list_table

from trashnet.entity.config_entity import (DataIngestionConfig, DataPreprocessingConfig)

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       custom_title_print,
                                       save_object,
                                       show_data_info,
                                       DataInspector)


class DataPreprocessing:
    def __init__(
            self, 
            data_ingestion_config: DataIngestionConfig = DataIngestionConfig,
            data_preprocessing_config: DataPreprocessingConfig = DataPreprocessingConfig
        ):
        
        try:
            self.data_ingestion_config = data_ingestion_config
            self.data_preprocessing_config = data_preprocessing_config

        except Exception as e:
           raise TrashClassificationException(e, sys)

    def initiate_data_preprocessing(self):
        """
        This method initiates the data preprocessing process.

        The method includes the following steps:

        1. Collecting images from the data download store directory
        2. Converting the image paths to a tf dataset
        3. Splitting the data into train and validation datasets
        4. Calculating the class distribution for the train and validation datasets
        5. Creating a label list table
        6. Converting the image paths to images
        7. Resizing the images
        8. Normalizing the images
        9. Augmenting the images
        10. Saving the train and validation datasets to tfrecord files
        11. Saving the class weights and label list to files

        Returns:
            DataPreprocessingConfig: The configuration used for the data preprocessing.
        """

        try: 
            tf.random.set_seed(self.data_preprocessing_config.seed)
            np.random.seed(self.data_preprocessing_config.seed)
            random.seed(self.data_preprocessing_config.seed)
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
            display_log_message(f"Collecting images from {color_text(str(self.data_ingestion_config.data_download_store_dir_path))}")
            all_images_paths = collect_and_combine_images(
                classes = self.data_preprocessing_config.label_list,
                train_path  = self.data_ingestion_config.data_download_store_dir_path,
                pattern_regex = self.data_preprocessing_config.image_extension_regex,
                seed=self.data_preprocessing_config.seed
            )

            display_log_message(f"Converting image paths to tf dataset")
            tf_paths = tf.data.Dataset.list_files(all_images_paths, shuffle=False)
            display_log_message(f'data: {color_text(tf_paths)}')
            display_log_message(f'number of data: {color_text(tf_paths.cardinality())}')

            display_log_message(f"Showing train files path info")
            file_info  = FilePathInfo(unit_file_size='KB')
            label_index = file_info.show_train_files_path_info(tf_paths, is_random=True)

            display_log_message("Splitting train and validation data")
            splitter = DatasetSplitter()
            train_tf_paths, valid_tf_paths = splitter.split_train_valid_test(
                dataset=tf_paths,
                split_ratio=self.data_preprocessing_config.split_ratio,
                shuffle=True,
                seed=self.data_preprocessing_config.seed
            )

            display_log_message(f"Showing class distribution")
            train_class_distribution, class_weights = calculate_class_distribution_tf(train_tf_paths, self.data_preprocessing_config.label_list)
            valid_class_distribution, _ = calculate_class_distribution_tf(valid_tf_paths, self.data_preprocessing_config.label_list)

            custom_title_print("Class distribution on Train set:")
            print_class_distribution(train_class_distribution)
            print()

            custom_title_print("Class distribution in Validation set:")
            print_class_distribution(valid_class_distribution)
            print()

            display_log_message(f"Creating label list table")
            label_table = create_label_list_table(self.data_preprocessing_config.label_list, default_value=-1)

            inspector = DataInspector(self.data_preprocessing_config.label_list)

            preprocessor = ImagePreprocessor(
                label_index=label_index,
                label_encoding=label_table,
                target_size=self.data_preprocessing_config.img_size
            )

            display_log_message(f"Converting images to tf dataset...")
            train_tf_images, valid_tf_images = preprocessor.convert_path_to_image(
                train_data=train_tf_paths,
                valid_data=valid_tf_paths,
            )

            display_log_message(f"Resizing images...")
            train_tf_images_resized, valid_tf_images_resized = preprocessor.image_resizing(
                train_data=train_tf_images,
                valid_data=valid_tf_images,
            )

            display_log_message(f"Normalizing images...")
            train_tf_images_normalized, valid_tf_images_normalized = preprocessor.image_normalization(
                train_data=train_tf_images_resized,
                valid_data=valid_tf_images_resized,
            )

            display_log_message(f"Augmenting images...")
            train_tf_images_augmented, valid_tf_images_augmented = preprocessor.image_augmentation(
                train_data=train_tf_images_normalized,
                valid_data=valid_tf_images_normalized
            )

            display_log_message(f"Showing data info...")
            show_data_info(
                    train_dataset=train_tf_images,
                    valid_dataset=valid_tf_images,
                )

            display_log_message(f"Inspecting data...")
            inspector.inspect(
                train_dataset=train_tf_images_augmented,
                valid_dataset=valid_tf_images_augmented
            )

            train_tf_images_augmented = train_tf_images_augmented.cache()
            valid_tf_images_augmented = valid_tf_images_augmented.cache()

            display_log_message(f"Saving train to {color_text(self.data_preprocessing_config.train_tfrecord_file_path)}...\n and validation dataset to {color_text(self.data_preprocessing_config.valid_tfrecord_file_path)}...")
            train_tf_images_augmented.save(str(self.data_preprocessing_config.train_tfrecord_file_path), compression="GZIP")
            valid_tf_images_augmented.save(str(self.data_preprocessing_config.valid_tfrecord_file_path), compression="GZIP")

            display_log_message(f"Saving class weights to {color_text(self.data_preprocessing_config.class_weights_file_path)}...")
            save_object(
                file_path=self.data_preprocessing_config.class_weights_file_path,
                obj=class_weights
            )

            display_log_message(f"Saving label list to {color_text(self.data_preprocessing_config.labels_list_file_path)}...")
            save_object(
                file_path=self.data_preprocessing_config.labels_list_file_path,
                obj=self.data_preprocessing_config.label_list
            )

            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            display_log_message(f"{class_name} config: {color_text(self.data_preprocessing_config)}")

            return self.data_preprocessing_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
