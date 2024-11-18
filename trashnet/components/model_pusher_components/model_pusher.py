import os
import sys
import inspect

from huggingface_hub import login
from dotenv import load_dotenv
from huggingface_hub import upload_file

from trashnet.exception import TrashClassificationException
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text)


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
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
            display_log_message("Pushing the model to Hugging Face Hub...")
            load_dotenv()  
            hf_token = os.getenv("HUGGINGFACE_TOKEN")
            login(hf_token)

            repo_id = self.model_pusher_config.repo_id
            best_model = str(self.model_trainer_config.model_file_path)

            upload_file(
                path_or_fileobj=best_model, 
                path_in_repo=self.model_pusher_config.model_file_path, 
                repo_id=repo_id,
                commit_message=self.model_pusher_config.commit_msg
            )

            display_log_message("Model pushed to Hugging Face Hub successfully!, open the link to view the model: https://huggingface.co/" + repo_id)

            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")
        
        except Exception as e:
            raise TrashClassificationException(e, sys)

