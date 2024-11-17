import tensorflow as tf
import os

class ImagePreprocessor:
    """
    Kelas ini bertanggung jawab untuk memproses gambar sebelum digunakan dalam pelatihan model.
    Ini mencakup konversi jalur file gambar menjadi gambar, mengubah ukuran gambar, normalisasi,
    dan augmentasi gambar untuk dataset pelatihan, validasi, dan pengujian.
    """

    def __init__(
            self, 
            label_index, 
            label_encoding, 
            target_size=(200, 200, 3), 
        ):

        """
        Inisialisasi kelas `ImagePreprocessor`.

        Args:
            label_index (int): Indeks jalur untuk mengambil label dari jalur gambar.
            label_encoding (tf.lookup.StaticHashTable): Objek lookup untuk mendapatkan label numerik dari nama kelas.
            target_size (tuple, optional): Ukuran target gambar yang diinginkan. Default adalah (200, 200).
            is_gray (bool, optional): Jika True, gambar akan dikonversi menjadi grayscale. Default adalah False.
        """
        self.label_index = label_index
        self.label_encoding = label_encoding
        self.target_size = target_size

    # ===================================================================================================
    # ------------------------------------ _CONVERT_PATH_TO_IMAGE_SINGLE --------------------------------
    # ===================================================================================================

    def _convert_path_to_image_single(
            self, 
            image_path
        ):

        """
        Converts image file paths to images and labels.
        Args:
            image_path (str): The path of the image file.
        Returns:
            Tuple: The converted image, numeric label, and image file path.
        """
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels= self.target_size[-1])
        image.set_shape([None, None,  self.target_size[-1]])
        split_img_path = tf.strings.split(image_path, os.path.sep)
        label = self.label_encoding.lookup(split_img_path[self.label_index])
        return image, label, image_path

    # ===================================================================================================
    # ------------------------------------ _RESIZE_IMAGE ------------------------------------------------
    # ===================================================================================================

    def _resize_image(
            self, 
            image, 
            label
        ):

        """
        Resizes the image to a predefined target size.
        Args:
            image (tf.Tensor): The input image.
            label (int): Image label.
        Returns:
            Tuple: The resized image and the label.
        """
        image = tf.image.resize(image, size=(self.target_size[0], self.target_size[1]))
        image = tf.cast(image, tf.uint8)
        return image, label

    # ===================================================================================================
    # ------------------------------------ _NORMALIZE_IMAGE ---------------------------------------------
    # ===================================================================================================

    def _normalize_image(
            self, 
            image, 
            label
        ):

        """
        Normalizes the image by dividing the pixel value by 255.
        Args:
            image (tf.Tensor): The input image.
            label (int): The label of the image.
        Returns:
            Tuple: Normalized image and label.
        """
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        label = tf.cast(label, tf.int32)
        return image, label

    # ===================================================================================================
    # ------------------------------------ _AUGMENT_IMAGE -----------------------------------------------
    # ===================================================================================================

    def _augment_image(
            self, 
            image, 
            label=None
        ):

        """
        Performs augmentation on images, including horizontal and vertical flipping and 90 or -90 degree rotation.
        Args:
            image (tf.Tensor): The input image.
            label (int, optional): The label of the image.
        Returns:
            Tuple: The augmented image and the label (if any).
        """

        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.clip_by_value(image, clip_value_min=0., clip_value_max=1.)
        return image, label


    # ===================================================================================================
    # ------------------------------------ _APPLY_CONVERT_PATH_TO_IMAGE ---------------------------------
    # ===================================================================================================

    def _apply_convert_path_to_image(
            self, 
            dataset
        ):
        """
        Performs file path to image conversion for the specified dataset.
        Args:
            dataset(tf.data.Dataset): Image file path dataset.
        Returns:
            tf.data.Dataset: The converted image dataset of the file path.
        """
        return dataset.map(
            map_func=lambda image_path: self._convert_path_to_image_single(image_path),
            num_parallel_calls=tf.data.AUTOTUNE
        ).cache()

    # ===================================================================================================
    # ------------------------------------ _APPLY_IMAGE_RESIZING ----------------------------------------
    # ===================================================================================================

    def _apply_image_resizing(
            self, 
            dataset
        ):

        """
        Resize the image in a specific dataset.
        Args:
            dataset (tf.data.Dataset): The image dataset.
        Returns:
            tf.data.Dataset: The resized image dataset.
        """
        return dataset.map(
            map_func=lambda image, label, path: self._resize_image(image, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )


    # ===================================================================================================
    # ------------------------------------ _APPLY_IMAGE_NORMALIZATION -----------------------------------
    # ===================================================================================================

    def _apply_image_normalization(
            self, 
            dataset
        ):

        """
        Performs image normalization on the specified dataset.
        Args:
            dataset(tf.data.Dataset): The image dataset.
        Returns:
            tf.data.Dataset: The normalized image dataset.
        """
        return dataset.map(
            map_func=lambda image, label: self._normalize_image(image, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # ===================================================================================================
    # ------------------------------------ _APPLY_IMAGE_AUGMENTATION ------------------------------------
    # ===================================================================================================

    def _apply_image_augmentation(
            self, 
            dataset
        ):

        """
        Performs image augmentation on the specified dataset.
        Args:
            dataset(tf.data.Dataset): The image dataset.
        Returns:
            tf.data.Dataset: The image dataset that has been augmented.
        """
        return dataset.map(
            map_func=lambda image, label: self._augment_image(image, label),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    # ===================================================================================================
    # ------------------------------------ CONVERT_PATH_TO_IMAGE ----------------------------------------
    # ===================================================================================================
    def convert_path_to_image(
            self, 
            **datasets
        ):

        """
        """
        return (self._apply_convert_path_to_image(dataset) for dataset in datasets.values())


    # ===================================================================================================
    # ----------------------------------------- IMAGE_RESIZING ------------------------------------------
    # ===================================================================================================
    def image_resizing(
            self, 
            **datasets
        ):

        """
        """
        return (self._apply_image_resizing(dataset) for dataset in datasets.values())


    # ===================================================================================================
    # ------------------------------------ IMAGE_NORMALIZATION ------------------------------------------
    # ===================================================================================================
    def image_normalization(
            self, 
            **datasets
        ):

        """
        """
        return (self._apply_image_normalization(dataset) for dataset in datasets.values())

    # ===================================================================================================
    # ------------------------------------ IMAGE_AUGMENTATION--------------------------------------------
    # ===================================================================================================
    def image_augmentation(
            self, 
            **datasets
        ):

        """
        """
        return (self._apply_image_augmentation(dataset) for dataset in datasets.values())