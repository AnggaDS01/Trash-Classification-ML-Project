import sys
import inspect 
import tensorflow as tf
import numpy as np
import random

from trashnet.exception import TrashClassificationException
from trashnet.ml.image_processor import ImagePreprocessor
from trashnet.utils.data_transformation_components_utils import *

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       custom_title_print,
                                       save_object,
                                       paths_exist,
                                       show_data_info,
                                       DataInspector)


class DataTransformation:
    def __init__(
            self, 
            data_ingestion_config,
            data_transformation_config
        ):
        
        try:
            self.data_ingestion_config = data_ingestion_config
            self.data_transformation_config = data_transformation_config

        except Exception as e:
           raise TrashClassificationException(e, sys)

    def initiate_data_transformation(self):
        tf.random.set_seed(self.data_transformation_config.seed)
        np.random.seed(self.data_transformation_config.seed)
        random.seed(self.data_transformation_config.seed)

        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('DataTransformation')} class in {color_text(file_name_function)}")

        try: 

            required_files = [
                self.data_transformation_config.train_tfrecord_data_path,
                self.data_transformation_config.valid_tfrecord_data_path,
                self.data_transformation_config.label_list_path,
                self.data_transformation_config.class_weights_path
            ]

            if paths_exist(required_files):
                display_log_message("All required files and folders are present. Skipping the process.")

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('DataTransformation')} class in {color_text(file_name_function)}")

                display_log_message(f"Data transformation config: {color_text(self.data_transformation_config)}")

                return self.data_transformation_config
            
            else:
                display_log_message("One or more required files/folders are missing. Running the process...")
                display_log_message(f"Collecting images from {color_text(str(self.data_ingestion_config.train_dir_path))}")
                all_images_paths = collect_and_combine_images(
                    classes = self.data_transformation_config.label_list,
                    train_path  = self.data_ingestion_config.train_dir_path,
                    pattern_regex = self.data_transformation_config.img_ext_regex_pattern,
                    seed=self.data_transformation_config.seed
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
                    split_ratio=self.data_transformation_config.split_ratio,
                    shuffle=True,
                    seed=self.data_transformation_config.seed
                )

                display_log_message(f"Showing class distribution")
                train_class_distribution, class_weights = calculate_class_distribution_tf(train_tf_paths, self.data_transformation_config.label_list)
                valid_class_distribution, _ = calculate_class_distribution_tf(valid_tf_paths, self.data_transformation_config.label_list)

                custom_title_print("Class distribution on Train set:")
                print_class_distribution(train_class_distribution)
                print()

                custom_title_print("Class distribution in Validation set:")
                print_class_distribution(valid_class_distribution)
                print()

                display_log_message(f"creating label list table")
                label_table = create_label_list_table(self.data_transformation_config.label_list, default_value=-1)

                inspector = DataInspector(self.data_transformation_config.label_list)

                preprocessor = ImagePreprocessor(
                    label_index=label_index,
                    label_encoding=label_table,
                    target_size=self.data_transformation_config.img_size
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

                display_log_message(f"Saving train to {color_text(self.data_transformation_config.train_tfrecord_data_path)}...\n and validation dataset to {color_text(self.data_transformation_config.valid_tfrecord_data_path)}...")
                train_tf_images_augmented.save(str(self.data_transformation_config.train_tfrecord_data_path), compression="GZIP")
                valid_tf_images_augmented.save(str(self.data_transformation_config.valid_tfrecord_data_path), compression="GZIP")

                display_log_message(f"Saving class weights to {color_text(self.data_transformation_config.class_weights_path)}...")
                save_object(
                    file_path=self.data_transformation_config.class_weights_path,
                    obj=class_weights
                )

                display_log_message(f"Saving label list to {color_text(self.data_transformation_config.label_list_path)}...")
                save_object(
                    file_path=self.data_transformation_config.label_list_path,
                    obj=self.data_transformation_config.label_list
                )

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('DataTransformation')} class in {color_text(file_name_function)}")

                display_log_message(f"Data transformation config: {color_text(self.data_transformation_config)}")

                return self.data_transformation_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
