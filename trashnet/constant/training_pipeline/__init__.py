"""
Directory:
"""
ARTIFACTS_DIR: str= "artifacts"
DATA_INGESTION_ARTIFACTS_DIR: str="data_ingestion"
DATA_TRAIN_DIR: str = "dataset-resized"
DATA_TRANSFORMATION_ARTIFACTS_DIR_PATH: str = "data_transformation"
OBJECT_DIR: str = "objects"
REPORT_DIR: str = "reports"
MODEL_DIR: str = "models"

"""
File:
"""
TRAIN_TFRECOARD_FILE: str = "train_trashnet.tfrecord"
VALID_TFRECORD_FILE: str = "valid_trashnet.tfrecord"
LABEL_LIST_FILE: str = "label_list.pkl"
CLASS_WEIGHTS_FILE: str = "class_weights.pkl"
KERAS_MODEL_FILE: str = "_model.keras"
TABEL_TRAINING_FILE: str = "tabel_pelatihan.csv"
TABEL_EPOCH_FILE: str = "tabel_epoch.csv"
PLOT_TRAINING_FILE: str = "plot_pelatihan.png"
PLOT_CONFUSION_MATRIX_FILE: str = "plot_confusion_matrix.png"
CLASSIFICATION_REPORT_FILE: str = "classification_report.txt"

"""
URL:
"""
DATA_DOWNLOAD_URL: str = "https://huggingface.co/datasets/garythung/trashnet/resolve/main/dataset-resized.zip"

"""
variables:
"""
MODEL_NAME: str = "DenseNet121"
PATTERN_IMAGE_EXT_REGEX: str = r"\.(jpe?g|png)$"
LABEL_LIST: list = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
SPLIT_RATIO: tuple = (0.9, 0.1)
IMAGE_SIZE: tuple = (224, 224)
WANDB_PROJECT_NAME: str = "trashnet-adatama-test-01"
LEARNING_RATE: float = 0.0001
LOSS_FUNCTION: str = "sparse_categorical_crossentropy"
METRICS: list = ['accuracy']
BATCH_SIZE: int = 32
EPOCHS: int = 150
DATASET: str = "TrashNet"