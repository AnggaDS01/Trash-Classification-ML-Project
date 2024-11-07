from dataclasses import dataclass
from pathlib import Path

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

@dataclass
class ModelTrainerConfig:
    model_dir_path: Path 
    model_path: Path
    report_dir_path: Path 
    tabel_training_path: Path
    tabel_epoch_path: Path
    plot_training_path: Path
    plot_confusion_matrix_path: Path
    classification_report_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    loss_function: str
    metrics: list
 

@dataclass
class WandbConfig:
    project: str
    config: dict