from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    data_ingestion_dir_path: Path

@dataclass
class DataTransformationArtifact:
    train_tfrecord_data_path: Path
    valid_tfrecord_data_path: Path
    label_list_path: Path
    class_weights_path: Path

@dataclass
class ModelTrainerArtifact:
    model_path: Path