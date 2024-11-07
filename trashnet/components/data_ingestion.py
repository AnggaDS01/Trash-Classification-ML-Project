import sys
import inspect

from colorama import Fore
from trashnet.exception import TrashClassificationException
from trashnet.utils.data_ingestion_components_utils import *

from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       paths_exist)

class DataIngestion:
    def __init__(
            self, 
            data_ingestion_config
        ):

        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
           raise TrashClassificationException(e, sys)

    def initiate_data_ingestion(self):
        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('DataIngestion')} class in {color_text(file_name_function)}")

        try: 
            required_files = [
                 self.data_ingestion_config.data_ingestion_dir_path
            ]
            
            if paths_exist(required_files):
                display_log_message("All required files and folders are present. Skipping the process.")

                display_log_message(f"Exited {color_text(function_name)} method of {color_text('DataIngestion')} class in {color_text(file_name_function)}")
                
                display_log_message(f"Data ingestion config: {color_text(self.data_ingestion_config)}")

                return self.data_ingestion_config
            
            else:
                display_log_message(f"Getting the data from URL: {color_text(self.data_ingestion_config.data_download_url, color=Fore.GREEN)}")

                download_and_extract_zip(
                    url=self.data_ingestion_config.data_download_url,
                    save_dir_path=self.data_ingestion_config.data_ingestion_dir_path,
                    extract_dir_path=self.data_ingestion_config.data_ingestion_dir_path,
                    is_file_removed=True
                )

                display_log_message(f"Got the data from URL: {color_text(self.data_ingestion_config.data_download_url, color=Fore.GREEN)}")
                display_log_message(f"Exited {color_text(function_name)} method of {color_text('DataIngestion')} class in {color_text(file_name_function)}")
                display_log_message(f"Data ingestion config: {color_text(self.data_ingestion_config)}")

                return self.data_ingestion_config

        except Exception as e:
            raise TrashClassificationException(e, sys)
