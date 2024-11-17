import sys
import re 
import random 

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from trashnet.exception import TrashClassificationException

from trashnet.utils.main_utils import (color_text, custom_title_print)

def collect_and_combine_images(
        classes, 
        train_path=None, 
        valid_path=None,
        test_path=None, 
        pattern_regex=r"\.(jpe?g)$", 
        num_images_per_class=None, 
        seed=42
    ):
    """
    Collects and merges images from the training and validation folders, and retrieves a random number of images from each class.
    Args:
        classes (list): List of classes (folder names) to process.
        train_path (str): The main path of the training folder that contains image data sub-folders.
        valid_path (str): The main path of the validation folder that contains the image data sub-folders.
        pattern_regex (str): The regex pattern for matching image file extensions (e.g. r'\.(jpg|png|jpeg)$').
        num_images_per_class (dict): Dictionary containing the number of images to fetch for each class. If None, all images will be retrieved.
        seed (int): Seed for random image retrieval. Default is 42.
    Returns:
        list: A combined list of image paths from the training and validation folders that were randomly picked.
    """

    try:
        def process_class(cls):
            # Combine images from training and validation for each class
            all_train_images = train_images_paths.get(cls, [])
            all_valid_images = valid_images_paths.get(cls, [])
            all_test_images = test_images_paths.get(cls, [])
            all_combined_images = all_train_images + all_valid_images + all_test_images

            # Retrieve a random number of images from the combined image
            return get_random_images(
                image_paths=all_combined_images,
                num_samples=None if num_images_per_class is None else num_images_per_class.get(cls, len(all_combined_images)),
                seed=seed
            )

        custom_title_print(f"COLLECT {classes} FROM TRAINING DATA")
        train_images_paths = collect_images_with_regex_and_count(train_path, classes, pattern_regex)
        custom_title_print(f"=")
        print()

        # Print the title for the image collection process of the validation data
        custom_title_print(f"COLLECT {classes} FROM VALIDATION DATA")
        valid_images_paths = collect_images_with_regex_and_count(valid_path, classes, pattern_regex)
        custom_title_print(f"=")
        print()

        # Print the title for the image collection process of the test data
        custom_title_print(f"COLLECT {classes} FROM TEST DATA")
        test_images_paths = collect_images_with_regex_and_count(test_path, classes, pattern_regex)
        custom_title_print(f"=")
        print()

        # Print titles for the process of merging images from training and validation
        custom_title_print(f"COMBINING {classes} FROM TRAINING AND VALIDATION DATA")

        random_images = {}

        # Using ThreadPoolExecutor to speed up the process of fetching images from each class in parallel
        with ThreadPoolExecutor() as executor:
            results = executor.map(process_class, classes)

        # Store the random image results for each class into the dictionary
        for cls, images in zip(classes, results):
            random_images[cls] = images
            print(f"Total {cls} taken: {color_text(len(random_images[cls]))}")

        # Merge all image paths from all classes
        all_images_paths = sum(random_images.values(), [])
        all_images_paths = [str(path) for path in all_images_paths]
        custom_title_print(f"Total images taken: {len(all_images_paths)}")

        return all_images_paths

    except Exception as e:
        raise TrashClassificationException(e, sys)
    

def collect_images_with_regex_and_count(
        path, 
        folders, 
        extensions_pattern
    ):

    """
    Collects image paths from multiple folders contained in the main path by filtering images
    by extension using regex. This function returns a dictionary with a key of
    folder name and the value is a list of image file paths.
    Args:
        path (str): The main path of the folder containing the image data sub-folders.
        folders (list): A list of folder names to scan for images.
        extensions_pattern (str): A regex pattern to match image file extensions (example: r'\.(jpg|png|jpeg)$').
    Returns:
        dict: Dictionary with the key being the folder name and the value being a list of image file paths that match the pattern.
    """

    try:
        image_paths = {folder: [] for folder in folders}   # Create an empty dictionary to store image paths
        pattern = re.compile(extensions_pattern, re.IGNORECASE) # Create a regex pattern to match image extensions with ignore case

        for folder in folders:
            folder_path = Path(path) / folder # Get the folder path

            for file_path in tqdm(folder_path.rglob("*"), desc=f"Collecting from {folder}", unit=" paths"):
                if pattern.search(file_path.suffix):  # Checks if the file extension matches the pattern
                    image_paths[folder].append(file_path)  # Add file paths to the list if they match

        return image_paths  # Returns the image path dictionary

    except Exception as e:
        print(f"No classes are retrieved from directory validation")
        return {}

def get_random_images(
        image_paths, 
        num_samples, 
        seed=42
    ):
    """
    Retrieves a random number of images from the image path list.
    Args:
        image_paths (list): A list of image paths.
        num_samples (int): The number of images to retrieve. If None, all images will be selected.
        seed (int): Seed to control the random retrieval results so that the results can be reproduced. Default is 42.
    Returns:
        list: A list of randomly selected image paths.
    """

    try:
        random.seed(seed)
        return random.sample(image_paths, min(len(image_paths) if num_samples is None else num_samples, len(image_paths)))

    except Exception as e:
        raise TrashClassificationException(e, sys)