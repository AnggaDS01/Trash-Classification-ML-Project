import sys
import inspect

from colorama import Fore

from trashnet.components.data_ingestion_components.utils.download_file import download_zip
from trashnet.components.data_ingestion_components.utils.extract_file import extract_zip
from trashnet.exception import TrashClassificationException
from trashnet.entity.config_entity import DataIngestionConfig

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text)

class DataIngestion:
    def __init__(
            self, 
            data_ingestion_config: DataIngestionConfig = DataIngestionConfig
        ):

        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
           raise TrashClassificationException(e, sys)

    def initiate_data_ingestion(self):
        try:

            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
            display_log_message(f"Getting the data from URL: {color_text(self.data_ingestion_config.data_download_url, color=Fore.GREEN)}")

            download_zip(
                url=self.data_ingestion_config.data_download_url,
                save_zip_file_path=self.data_ingestion_config.zip_file_path,
            )

            extract_zip(
                zip_file_path=self.data_ingestion_config.zip_file_path,
                extract_dir=self.data_ingestion_config.data_ingestion_dir_path
            )

            display_log_message(f"Got the data from URL: {color_text(self.data_ingestion_config.data_download_url, color=Fore.GREEN)}")
            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
            display_log_message(f"{class_name} config: {color_text(self.data_ingestion_config)}")

            return self.data_ingestion_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
