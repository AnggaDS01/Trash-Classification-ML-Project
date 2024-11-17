import os

from trashnet.constant import *
from pathlib import Path
from datetime import datetime
from trashnet.utils.main_utils import read_yaml, create_directories
from trashnet.entity.config_entity import (DataIngestionConfig,
                                           DataTransformationConfig,
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

        create_directories([self.config.artifacts_root_dir])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.data_ingestion_dir_path])

        data_ingestion_config = DataIngestionConfig(
            data_ingestion_dir_path = config.data_ingestion_dir_path,
            data_download_store_dir_path = config.data_download_store_dir_path,
            zip_file_path = config.zip_file_path,
            data_download_url = config.data_download_url,
        )

        return data_ingestion_config
    



    # def get_data_transformation_config(self) -> DataTransformationConfig:
    #     data_transformation_dir_path = Path(os.path.join(self.artifacts_config, self.directories_config.DATA_TRANSFORMATION))
    #     train_tfrecord_data_path = data_transformation_dir_path / self.tfrecords_config.TRAIN_FILE
    #     valid_tfrecord_data_path = data_transformation_dir_path / self.tfrecords_config.VALID_FILE
        
    #     object_dir_path = Path(os.path.join(self.artifacts_config, self.directories_config.OBJECTS))
    #     label_list_path = object_dir_path / self.objects_config.LABEL_LIST_FILE
    #     class_weights_path = object_dir_path / self.objects_config.CLASS_WEIGHTS_FILE

    #     img_ext_regex_pattern = self.data_config.IMAGE_EXTENSION_REGEX
    #     label_list = self.dataset_params.LABEL_LIST
    #     split_ratio = self.dataset_params.SPLIT_RATIO
    #     img_size = self.model_params.IMAGE_SIZE
    #     seed = self.training_params.SEED

    #     data_transformation_config = DataTransformationConfig(
    #         data_transformation_dir_path = data_transformation_dir_path,
    #         train_tfrecord_data_path = train_tfrecord_data_path,
    #         valid_tfrecord_data_path = valid_tfrecord_data_path,
    #         object_dir_path = object_dir_path,
    #         label_list_path = label_list_path,
    #         class_weights_path = class_weights_path,
    #         img_ext_regex_pattern = img_ext_regex_pattern,
    #         label_list = label_list,
    #         split_ratio = split_ratio,
    #         img_size = img_size,
    #         seed = seed
    #     )

    #     return data_transformation_config



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

# if __name__ == '__main__':
#     config = ConfigurationManager()
#     get_config = config.get_data_ingestion_config()

#     print(get_config)