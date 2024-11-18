import sys
import inspect 
import tensorflow as tf

from trashnet.exception import TrashClassificationException
from trashnet.components.prediction_components.utils.predict_image import predict_image
from trashnet.components.prediction_components.utils.prediction_data_preprocessing import ImagePreprocessor
from trashnet.utils.main_utils import (display_log_message, 
                                       display_function_info,
                                       color_text,
                                       load_object,
                                       custom_title_print)


class Prediction:
    def __init__(
            self,
            model_trainer_config, 
            data_preprocessing_config,
        ):

        try:
            self.model_trainer_config = model_trainer_config
            self.data_preprocessing_config = data_preprocessing_config
        except Exception as e:
            raise TrashClassificationException(e, sys)
        
    def initiate_prediction(self, img_path, augment):

        try:
            function_name, class_name, file_name = display_function_info(inspect.currentframe())
            display_log_message(f"Started {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            display_log_message(f"Loading label list from file: {color_text(self.data_preprocessing_config.labels_list_file_path)}")
            label_list = load_object(file_path=self.data_preprocessing_config.labels_list_file_path)

            display_log_message(f"Loading the best model from {color_text(self.model_trainer_config.model_file_path)}")
            best_model = tf.keras.models.load_model(self.model_trainer_config.model_file_path)

            preprocessor = ImagePreprocessor(target_size=(self.data_preprocessing_config.img_size))
            processed_image = preprocessor.prepare_for_model(img_path, augment)
            predicted_label = predict_image(best_model, processed_image, label_list)

            # Display exit message
            display_log_message(f"Exited {color_text(function_name)} method of {color_text(class_name)} class in {color_text(file_name)}")

            return predicted_label

        except Exception as e:
            raise TrashClassificationException(e, sys)
