import sys
import inspect

from PIL import Image
from matplotlib import pyplot as plt
from trashnet.exception import TrashClassificationException
from trashnet.components.prediction_components.prediction import Prediction
from trashnet.configuration.configuration import ConfigurationManager


from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text)

class PredictionPipeline:

    def __init__(self):
        config = ConfigurationManager()
        self.model_trainer_config = config.get_model_trainer_config()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
    
    def start_prediction(
            self,
            img_path: str,
            augment: bool
        ) -> str:
        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Entered {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            prediction = Prediction(
                model_trainer_config = self.model_trainer_config,
                data_preprocessing_config = self.data_preprocessing_config
            )

            prediction_result = prediction.initiate_prediction(img_path, augment)

            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            return prediction_result

        except Exception as e:
            raise TrashClassificationException(e, sys)
        

    def run_pipeline(
            self, 
            img_path=None,
            augment=None
        ) -> None:

        try:
            prediction_result = self.start_prediction(img_path, augment)
            return prediction_result
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
        
if __name__ == "__main__":
    img_path = 'artifacts/data_ingestion/dataset-resized/cardboard/cardboard20.jpg'
    image = Image.open(img_path) 
    
    prediction_pipeline = PredictionPipeline()
    prediction_result = prediction_pipeline.run_pipeline(img_path)

    plt.figure(figsize=(12, 8))
    plt.imshow(image)
    plt.title(f"Predicted Label: {prediction_result}")
    plt.axis('off')
    plt.show()