import tensorflow as tf

from trashnet.constant.training_pipeline import *

base_model = tf.keras.applications.DenseNet121(
    weights='imagenet',  # Menggunakan bobot dari pretrained model di ImageNet
    include_top=False,   # Tidak menyertakan lapisan fully connected atas agar bisa menyesuaikan dataset kita
    input_shape=(*IMAGE_SIZE, 3)  # Menyesuaikan input shape dengan ukuran gambar kita
)

def build_model(
      input_shape=None,
      num_classes=None,
      pretrained_model=None,
    ) -> tf.keras.Model:
    # Input layer
    input_layer = tf.keras.Input(shape=input_shape)

    # Base model pretrained
    pretrained_model.trainable = False

    # Input melewati base model
    x = pretrained_model(input_layer, training=False)

    # Global average pooling layer
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Output layer dengan aktivasi sigmoid
    output_layer = tf.keras.layers.Dense(1 if len(num_classes) == 2 else len(num_classes), activation='softmax')(x)

    # Membuat model
    model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

    return model