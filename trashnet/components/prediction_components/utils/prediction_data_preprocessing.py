import tensorflow as tf

class ImagePreprocessor:
    def __init__(
            self, 
            target_size=(200, 200, 3), 
        ):

        """Initialize the image preprocessor with the target size."""
        self.target_size = target_size

    def _resize_image(self, image):
        """Resize input image to target size."""
        return tf.image.resize(image, size=(self.target_size[0], self.target_size[1]))

    def _normalize_image(self, image):
        """Normalize the image to the range [0, 1]."""
        image = tf.cast(image, tf.float32)
        image = image / 255.0
        return image

    def _augment_image(self, image):
        """Apply augmentation to the image (e.g. flipping, rotation)."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        return image

    def preprocess(self, image, augment=False):
        """
        Perform preprocessing on the image, including resizing, normalization, and optional augmentation.

        Args:
        image: Input image.
        augment: Flag to enable augmentation.

        Returns:
        Preprocessed image.
        """
        image = self._resize_image(image)
        if augment:
            image = self._augment_image(image)
        image = self._normalize_image(image)
        return image

    def prepare_for_model(self, image_input, augment=False):
        """
        Prepare images for model input, either batch or single image.

        Args:
        image_input: Path to image or image Tensor.
        augment: Flag to enable augmentation.

        Returns:
        Preprocessed image ready for model.
        """
        if isinstance(image_input, str):
            image = tf.io.read_file(image_input)
            image = tf.image.decode_image(image, channels=self.target_size[-1])
        else:
            image = image_input

        image = self.preprocess(image, augment)
        image = tf.expand_dims(image, axis=0) if len(image.shape) == 3 else image
        return image

    def preprocess_batch(self, image_batch, augment=False):
        """
        Preprocessing for image batches, suitable for batch prediction or training.

        Args:
        image_batch: Batch of images in Tensor form.
        augment: Flag to enable augmentation.

        Returns:
        Batch of preprocessed images.
        """
        image_batch = tf.map_fn(lambda img: self.preprocess(img, augment), image_batch)
        return image_batch