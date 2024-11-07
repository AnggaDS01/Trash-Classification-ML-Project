from dataclasses import dataclass
from pathlib import Path
# from trashnet.constant.training_pipeline import *

@dataclass
class DataIngestionConfig:
    data_ingestion_dir_path: Path
    train_dir_path: Path
    data_download_url: str

@dataclass
class DataTransformationConfig:
    data_transformation_dir_path: Path
    train_tfrecord_data_path: Path
    valid_tfrecord_data_path: Path
    object_dir_path: Path
    label_list_path: Path
    class_weights_path: Path

    img_ext_regex_pattern: str
    label_list: list
    split_ratio: tuple
    img_size: tuple























# @dataclass
# class TrainingPipelineConfig:
#     artifacts_dir: str = ARTIFACTS_DIR

# training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig() 


# @dataclass
# class DataIngestionConfig:
#     data_ingestion_dir_path: Path = Path(os.path.join(
#         training_pipeline_config.artifacts_dir, DATA_INGESTION_ARTIFACTS_DIR
#     ))

#     data_download_url: str = DATA_DOWNLOAD_URL

# @dataclass
# class DataTransformationConfig:
#     data_transformation_dir_path: Path = Path(os.path.join(
#         training_pipeline_config.artifacts_dir, DATA_TRANSFORMATION_ARTIFACTS_DIR_PATH
#     ))

#     train_tfrecord_data_path: Path = Path(os.path.join(
#         data_transformation_dir_path, TRAIN_TFRECOARD_FILE
#     ))

#     valid_tfrecord_data_path: Path = Path(os.path.join(
#         data_transformation_dir_path, VALID_TFRECORD_FILE
#     ))

#     object_dir_path: Path = Path(os.path.join(
#         training_pipeline_config.artifacts_dir, OBJECT_DIR
#     ))

#     label_list_path: Path = Path(os.path.join(
#         object_dir_path, LABEL_LIST_FILE
#     ))

#     class_weights_path: Path = Path(os.path.join(
#         object_dir_path, CLASS_WEIGHTS_FILE
#     ))


# @dataclass
# class ModelTrainerConfig:
#     model_dir_path: Path = Path(os.path.join(
#         training_pipeline_config.artifacts_dir, MODEL_DIR
#     ))

#     model_path: Path = Path(os.path.join(
#         model_dir_path, MODEL_NAME + KERAS_MODEL_FILE
#     ))

#     report_dir_path: Path = Path(os.path.join(
#         training_pipeline_config.artifacts_dir, REPORT_DIR
#     ))

#     tabel_training_path: Path = Path(os.path.join(
#         report_dir_path, TABEL_TRAINING_FILE
#     ))

#     tabel_epoch_path: Path = Path(os.path.join(
#         report_dir_path, TABEL_EPOCH_FILE
#     ))

#     plot_training_path: Path = Path(os.path.join(
#         report_dir_path, PLOT_TRAINING_FILE
#     ))

#     plot_confusion_matrix_path: Path = Path(os.path.join(
#         report_dir_path, PLOT_CONFUSION_MATRIX_FILE
#     ))

#     classification_report_path: Path = Path(os.path.join(
#         report_dir_path, CLASSIFICATION_REPORT_FILE
#     ))    