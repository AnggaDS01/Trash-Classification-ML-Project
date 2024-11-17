from trashnet.utils.main_utils import custom_title_print

class DatasetSplitter:
    """
    This class is used to split the dataset into three parts: training, validation, and testing.
    The division is done based on a configurable ratio, and the dataset can be randomized before division.
    """

    def __init__(self):
        """
        Initialize the `DatasetSplitter` class. No arguments are accepted during initialization.
        """
        pass

    def split_train_valid_test(self, dataset, split_ratio=None, shuffle=True, buffer_size=None, seed=42):
        """
        Splits the dataset into three parts: training, validation, and testing.
        Args:
            dataset(tf.data.Dataset): The dataset to be split.
            train_ratio (float, optional): The ratio of data for training. Default is 0.7.
            valid_ratio (float, optional): Data ratio for validation. Default is 0.2.
            shuffle (bool, optional): Whether the dataset needs to be shuffled before division. Default is True.
            buffer_size (int, optional): The buffer size for dataset shuffling. If None, buffer_size is taken from the dataset size.
            seed (int, optional): Seed for dataset randomization. Default is 42.
        Returns:
            tuple: A tuple containing the three split datasets: (train_dataset, val_dataset, test_dataset).
        """

        try:
            dataset_size = len(dataset) if buffer_size is None else buffer_size

            # Define individual ratios for training, validation, and test sets
            train_ratio = split_ratio[0]
            val_ratio = split_ratio[1]
            test_ratio = round(max(1.0 - (train_ratio + val_ratio), 0), 4)  # Calculate remaining ratio for test set

            # Verify the total ratio equals 1.0; raise an error if not
            total_ratio = round(sum((train_ratio, val_ratio, test_ratio)), 2)
            if total_ratio != 1.0:
                raise ValueError("[ERROR] split_ratio must sum to 1.0.\n")

            # Determine the number of images in each split based on the calculated ratios
            train_size = int(round(dataset_size * train_ratio, 0))
            val_size = int(round(dataset_size * val_ratio, 0))
            test_size = int(round(dataset_size * test_ratio, 0))

            # Randomly shuffle the image files if random_split is enabled
            if shuffle:
                dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed)

            # Split the files into training, validation, and test sets based on calculated sizes
            train_dataset = dataset.take(train_size)
            val_test_dataset = dataset.skip(train_size)

            # If test_size == 0, only create train and validation datasets
            if test_size == 0:
                val_dataset = val_test_dataset.take(val_size)
                self._display_info(
                    dataset=dataset,
                    train_dataset=train_dataset,
                    valid_dataset=val_dataset,
                    dataset_size=dataset_size,
                    shuffle=shuffle,
                    test_size=test_size
                )

                return train_dataset, val_dataset
            else:
                val_dataset = val_test_dataset.take(val_size)
                test_dataset = val_test_dataset.skip(val_size)

                self._display_info(
                    dataset=dataset,
                    train_dataset=train_dataset,
                    valid_dataset=val_dataset,
                    test_dataset=test_dataset,
                    dataset_size=dataset_size,
                    shuffle=shuffle,
                    test_size=test_size
                )

                return train_dataset, val_dataset, test_dataset

        except Exception as e:
            print(f"Error: {e}")
            return None, None, None

    def _display_info(self, dataset=None, train_dataset=None, valid_dataset=None, test_dataset=None, dataset_size=None, shuffle=False, test_size=None):
        """
        Displays information about dataset splits such as size, ratio, and shuffle status.
        Args:
            dataset(tf.data.Dataset): The original dataset that has not been split.
            train_dataset (tf.data.Dataset): The training part of the dataset.
            valid_dataset (tf.data.Dataset): The validation part of the dataset.
            test_dataset (tf.data.Dataset): The testing dataset.
            dataset_size (int): The total size of the dataset.
            shuffle (bool): State whether the dataset is shuffled before dividing.
        """
        train_ratio = len(train_dataset) / dataset_size
        valid_ratio = len(valid_dataset) / dataset_size
        test_ratio = len(test_dataset) / dataset_size if test_size > 0 else 0

        print(f"Total number of data: {dataset_size}")
        print(f"Shuffle status: {shuffle}")

        custom_title_print(' Training Dataset ')
        print(f"Info data: {train_dataset}")
        print(f"Training Split: {round(train_ratio * 100, 2)}%")
        print(f"Number of data: {len(train_dataset)}")
        print()

        custom_title_print(' Validation Dataset ')
        print(f"Info data: {valid_dataset}")
        print(f"Validation Split: {round(valid_ratio * 100, 2)}%")
        print(f"Number of data: {len(valid_dataset)}")
        print()

        if test_size > 0:
            custom_title_print(' Test Dataset ')
            print(f"Info data: {test_dataset}")
            print(f"Test Split: {round(test_ratio * 100, 2)}%")
            print(f"Number of data: {len(test_dataset)}")