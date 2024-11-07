import os

from trashnet.constant import *
from pathlib import Path
from trashnet.utils.main_utils import read_yaml
from trashnet.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        self.directories_config = self.config.DIRECTORIES
        self.data_config = self.config.DATA
        self.tfrecords_config = self.config.TFRECORDS
        self.model_config = self.config.MODEL
        self.objects_config = self.config.OBJECTS
        self.reports_config = self.config.REPORTS

        self.model_params = self.params.MODEL
        self.dataset_params = self.params.DATASET
        self.training_params = self.params.TRAINING
        self.wandb_params = self.params.WANDB
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_ingestion_dir_path = Path(os.path.join(self.directories_config.ARTIFACTS, self.directories_config.DATA_INGESTION))
        train_dir_path = data_ingestion_dir_path / self.data_config.TRAIN_DIR

        data_download_url = self.data_config.DOWNLOAD_URL

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir_path = data_ingestion_dir_path,
            train_dir_path = train_dir_path,
            data_download_url = data_download_url,
        )

        return data_ingestion_config
    




    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_transformation_dir_path = Path(os.path.join(self.directories_config.ARTIFACTS, self.directories_config.DATA_TRANSFORMATION))
        train_tfrecord_data_path = data_transformation_dir_path / self.tfrecords_config.TRAIN_FILE
        valid_tfrecord_data_path = data_transformation_dir_path / self.tfrecords_config.VALID_FILE
        
        object_dir_path = Path(os.path.join(self.directories_config.ARTIFACTS, self.directories_config.OBJECTS))
        label_list_path = object_dir_path / self.objects_config.LABEL_LIST_FILE
        class_weights_path = object_dir_path / self.objects_config.CLASS_WEIGHTS_FILE

        img_ext_regex_pattern = self.data_config.IMAGE_EXTENSION_REGEX
        label_list = self.dataset_params.LABEL_LIST
        split_ratio = self.dataset_params.SPLIT_RATIO
        img_size = self.model_params.IMAGE_SIZE

        data_transformation_config = DataTransformationConfig(
            data_transformation_dir_path = data_transformation_dir_path,
            train_tfrecord_data_path = train_tfrecord_data_path,
            valid_tfrecord_data_path = valid_tfrecord_data_path,
            object_dir_path = object_dir_path,
            label_list_path = label_list_path,
            class_weights_path = class_weights_path,
            img_ext_regex_pattern = img_ext_regex_pattern,
            label_list = label_list,
            split_ratio = split_ratio,
            img_size = img_size
        )

        return data_transformation_config

if __name__ == '__main__':
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_transformation_config()
#     print(data_ingestion_config.data_transformation_dir_path)
#     print(data_ingestion_config.train_tfrecord_data_path)
#     print(data_ingestion_config.valid_tfrecord_data_path)
#     print(data_ingestion_config.object_dir_path)
#     print(data_ingestion_config.label_list_path)
#     print(data_ingestion_config.class_weights_path)
#     print(data_ingestion_config.img_ext_regex_pattern)
#     print(data_ingestion_config.label_list)
#     print(data_ingestion_config.split_ratio)
    print(data_ingestion_config.img_size)