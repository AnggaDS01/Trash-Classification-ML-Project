from trashnet.constant import *
from pathlib import Path
from trashnet.utils.main_utils import read_yaml, create_directories
from trashnet.entity.config_entity import (DataIngestionConfig,
                                           DataPreprocessingConfig,
                                           ModelTrainerConfig,
                                           ModelEvaluationConfig,
                                           ModelPusher,
                                           WandbConfig)


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.ARTIFACTS_ROOT_DIR])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.DATA_INGESTION
        create_directories([config.DATA_INGESTION_DIR_PATH])

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir_path = Path(config.DATA_INGESTION_DIR_PATH),
            data_download_store_dir_path = Path(config.DATA_DOWNLOAD_STORE_DIR_PATH),
            zip_file_path = Path(config.ZIP_FILE_PATH),
            data_download_url = config.DATA_DOWNLOAD_URL,
        )

        return data_ingestion_config
    


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.DATA_PREPROCESSING
        
        create_directories([config.TFRECORDS_DIR_PATH])
        create_directories([config.OBJECTS_DIR_PATH])

        data_preprocessing_config = DataPreprocessingConfig(
            train_tfrecord_file_path = Path(config.TRAIN_TFRECORD_FILE_PATH),
            valid_tfrecord_file_path = Path(config.VALID_TFRECORD_FILE_PATH),
            labels_list_file_path = Path(config.LABELS_LIST_FILE_PATH),
            class_weights_file_path = Path(config.CLASS_WEIGHTS_FILE_PATH),
            image_extension_regex = config.IMAGE_EXTENSION_REGEX,
            label_list = self.params.LABEL_LIST,
            split_ratio = tuple(self.params.SPLIT_RATIO),
            img_size = self.params.IMAGE_SIZE,
            seed = self.params.SEED,
        )

        return data_preprocessing_config



    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.MODEL_TRAINING
        
        create_directories([config.MODEL_DIR_PATH])
        create_directories([config.REPORTS_DIR_PATH])

        model_trainer_config = ModelTrainerConfig(
            model_file_path = Path(config.MODEL_FILE_PATH),
            training_table_file_path = Path(config.TRAINING_TABLE_FILE_PATH),
            epoch_table_file_path = Path(config.EPOCH_TABLE_FILE_PATH),
            training_plot_file_path = Path(config.TRAINING_PLOT_FILE_PATH),
            batch_size = self.params.BATCH_SIZE,
            epochs = self.params.EPOCHS,
            learning_rate = self.params.LEARNING_RATE,
            loss_function = self.params.LOSS_FUNCTION,
            metrics = self.params.METRICS,
        )

        return model_trainer_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.MODEL_EVALUATION

        model_evaluation_config = ModelEvaluationConfig(
            plot_confusion_matrix_file_path = config.CONFUSION_MATRIX_FILE_PATH,
            classification_report_file_path = config.CLASSIFICATION_REPORT_FILE_PATH,
            normalize_confusion_matrix = self.params.NORMALIZE_CONFUSION_MATRIX,
            figsize = self.params.FIGSIZE,
        )

        return model_evaluation_config

    # def get_model_pusher_config(self) -> ModelPusher:
    #     repo_id = self.huggingface_config.REPO_ID
    #     commit_msg = self.huggingface_config.COMMIT_MSG

    #     model_evaluation_config = ModelPusher(
    #         repo_id= repo_id,
    #         commit_msg= commit_msg
    #     )

    #     return model_evaluation_config


    def get_wandb_config(self) -> WandbConfig:
        config = self.config.WANDB

        config_dicts = {
            "learning_rate": self.params.LEARNING_RATE,
            "loss_function": self.params.LOSS_FUNCTION,
            "metrics": self.params.METRICS,
            "batch_size": self.params.BATCH_SIZE,
            "epochs": self.params.EPOCHS,
            "architecture": self.params.MODEL_NAME,
            "dataset": self.params.DATASET_NAME
        }

        project = config.PROJECT_NAME
        sweep_config = config.SWEEP_CONFIG
        sweep_count = config.SWEEP_COUNT
        
        wandb_config = WandbConfig(
            project_name = project,
            config = config_dicts,
            sweep_config = sweep_config,
            sweep_count = sweep_count
        )

        return wandb_config

# if __name__ == '__main__':
#     config = ConfigurationManager()
#     get_config = config.get_model_evaluation_config()

#     print(get_config)