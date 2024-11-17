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



    # def get_model_trainer_config(self) -> ModelTrainerConfig:
    #     model_dir_path =  Path(os.path.join(self.artifacts_config, self.directories_config.MODELS))
    #     model_path = model_dir_path / (self.model_params.NAME + self.model_config.KERAS_FILE)

    #     report_dir_path = Path(os.path.join(self.artifacts_config, self.directories_config.REPORTS))
    #     training_tabel_path = report_dir_path / self.model_params.NAME / self.reports_config.TRAINING_TABEL
    #     tabel_epoch_path = report_dir_path / self.model_params.NAME / self.reports_config.EPOCH_TABEL
    #     plot_training_path = report_dir_path / self.model_params.NAME / self.reports_config.TRAINING_PLOT

    #     batch_size =  self.training_params.BATCH_SIZE
    #     epochs =  self.training_params.EPOCHS
    #     learning_rate = self.training_params.LEARNING_RATE
    #     loss_function = self.training_params.LOSS_FUNCTION
    #     metrics = self.training_params.METRICS


    #     model_trainer_config = ModelTrainerConfig(
    #         model_dir_path = model_dir_path,
    #         model_path = model_path,
    #         report_dir_path = report_dir_path,
    #         training_tabel_path = training_tabel_path,
    #         tabel_epoch_path = tabel_epoch_path,
    #         plot_training_path = plot_training_path,
    #         batch_size = batch_size,
    #         epochs = epochs,
    #         learning_rate = learning_rate,
    #         loss_function = loss_function,
    #         metrics = metrics
    #     )

    #     return model_trainer_config


    # def get_model_evaluation_config(self) -> ModelEvaluationConfig:
    #     report_dir_path = Path(os.path.join(self.artifacts_config, self.directories_config.REPORTS))

    #     plot_confusion_matrix_path = report_dir_path / self.model_params.NAME / self.reports_config.CONFUSION_MATRIX_PLOT
    #     classification_report_path = report_dir_path / self.model_params.NAME / self.reports_config.CLASSIFICATION_REPORT

    #     normalize = self.evaluation_params.NORMALIZE_CONFUSION_MATRIX
    #     figsize = tuple(self.evaluation_params.FIGSIZE)

    #     model_evaluation_config = ModelEvaluationConfig(
    #         plot_confusion_matrix_path = plot_confusion_matrix_path,
    #         classification_report_path = classification_report_path,
    #         normalize = normalize,
    #         figsize = figsize
    #     )

    #     return model_evaluation_config

    # def get_model_pusher_config(self) -> ModelPusher:
    #     repo_id = self.huggingface_config.REPO_ID
    #     commit_msg = self.huggingface_config.COMMIT_MSG

    #     model_evaluation_config = ModelPusher(
    #         repo_id= repo_id,
    #         commit_msg= commit_msg
    #     )

    #     return model_evaluation_config


    # def get_wandb_config(self) -> WandbConfig:
    #     config = {
    #         "learning_rate": self.training_params.LEARNING_RATE,
    #         "loss_function": self.training_params.LOSS_FUNCTION,
    #         "metrics": self.training_params.METRICS,
    #         "batch_size": self.training_params.BATCH_SIZE,
    #         "epochs": self.training_params.EPOCHS,
    #         "architecture": self.model_params.NAME,
    #         "dataset": self.dataset_params.NAME
    #     }

    #     project = self.wandb_params.PROJECT_NAME

    #     sweep_config = self.wandb_params.SWEEP_CONFIG

    #     sweep_count = self.wandb_params.SWEEP_COUNT
        
    #     wandb_config = WandbConfig(
    #         config = config,
    #         project = project,
    #         sweep_config = sweep_config,
    #         sweep_count = sweep_count
    #     )

    #     return wandb_config

if __name__ == '__main__':
    config = ConfigurationManager()
    get_config = config.get_data_preprocessing_config()

    print(get_config)