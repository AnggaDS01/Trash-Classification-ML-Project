import os
import sys
import inspect

from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import upload_file

from trashnet.exception import TrashClassificationException
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_name,
                                       color_text,
                                       )


class ModelPusher:
    def __init__(
            self,
            model_trainer_config,
            model_pusher_config
        ):

        try:
            self.model_trainer_config = model_trainer_config
            self.model_pusher_config = model_pusher_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_model_pusher(self):
        function_name, file_name_function = display_function_name(inspect.currentframe())
        display_log_message(f"Entered {color_text(function_name)} method of {color_text('ModelPusher')} class in {color_text(file_name_function)}")

        try:
            display_log_message("One or more required files/folders are missing. Running the process...")

            display_log_message("Pushing the model to Hugging Face Hub...")
            load_dotenv()  
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            login(hf_token)

            repo_id = self.model_pusher_config.repo_id
            local_model_path = str(self.model_trainer_config.model_path)

            upload_file(
                path_or_fileobj=local_model_path, 
                path_in_repo=local_model_path.split("/")[-1], 
                repo_id=repo_id,
                commit_message=self.model_pusher_config.commit_msg
            )

            display_log_message(f"Exited {color_text(function_name)} method of {color_text('ModelPusher')} class in {color_text(file_name_function)}")
            display_log_message(f"Model Pusher config: {color_text(self.model_evaluation_config)}")

            return self.model_evaluation_config

        except Exception as e:
            raise TrashClassificationException(e, sys)

