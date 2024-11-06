"""
Directory:
"""
ARTIFACTS_DIR: str= "artifacts"
DATA_INGESTION_ARTIFACTS_DIR: str="data_ingestion"
DATA_TRAIN_DIR: str = "dataset-resized"
DATA_TRANSFORMATION_ARTIFACTS_DIR_PATH: str = "data_transformation"
OBJECT_DIR: str = "objects"

"""
File:
"""
TRAIN_TFRECOARD_FILE: str = "train_trashnet.tfrecord"
VALID_TFRECORD_FILE: str = "valid_trashnet.tfrecord"
LABEL_LIST_FILE: str = "label_list.pkl"
CLASS_WEIGHTS_FILE: str = "class_weights.pkl"

"""
URL:
"""
DATA_DOWNLOAD_URL: str = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

"""
variables:
"""
PATTERN_IMAGE_EXT_REGEX: str = r"\.(jpe?g|png)$"
LABEL_LIST: list = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
SPLIT_RATIO: tuple = (0.9, 0.1)
IMAGE_SIZE: tuple = (224, 224)
