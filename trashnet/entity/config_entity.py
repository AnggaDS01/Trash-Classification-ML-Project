import os
from dataclasses import dataclass
from pathlib import Path
from trashnet.constant.training_pipeline import *

@dataclass
class TrainingPipelineConfig:
    artifacts_dir: str = ARTIFACTS_DIR

training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig() 



@dataclass
class DataIngestionConfig:
    data_ingestion_dir_path: Path = Path(os.path.join(
        training_pipeline_config.artifacts_dir, DATA_INGESTION_ARTIFACTS_DIR
    ))

    data_download_url: str = DATA_DOWNLOAD_URL



@dataclass
class DataTransformationConfig:
    data_transformation_dir_path: Path = Path(os.path.join(
        training_pipeline_config.artifacts_dir, DATA_TRANSFORMATION_ARTIFACTS_DIR_PATH
    ))

    train_tfrecord_data_path: Path = Path(os.path.join(
        data_transformation_dir_path, TRAIN_TFRECOARD_FILE
    ))

    valid_tfrecord_data_path: Path = Path(os.path.join(
        data_transformation_dir_path, VALID_TFRECORD_FILE
    ))

    object_dir_path: Path = Path(os.path.join(
        training_pipeline_config.artifacts_dir, OBJECT_DIR
    ))

    label_list_path: Path = Path(os.path.join(
        object_dir_path, LABEL_LIST_FILE
    ))

    class_weights_path: Path = Path(os.path.join(
        object_dir_path, CLASS_WEIGHTS_FILE
    ))