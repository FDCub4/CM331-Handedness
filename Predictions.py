
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
images = []

class ComplexResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_sizes, **kwargs):
        super().__init__(**kwargs)
        self.convs = [
            layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
            for kernel_size in kernel_sizes
        ]
        self.bns = [layers.BatchNormalization() for _ in kernel_sizes]
        self.shortcut = layers.Conv2D(filters, (1, 1), padding='same')  # Shortcut for residual connection

    def call(self, inputs):
        x = inputs
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
        shortcut = self.shortcut(inputs)  # Ensure the shortcut has the same number of filters
        return layers.add([x, shortcut])

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.convs[0].filters,
            'kernel_sizes': [conv.kernel_size for conv in self.convs]
        })
        return config


for filename in os.listdir(r"updated_images"):
    images.append(cv2.imread(r"updated_images/" + filename)[897:1153 ,101:2149])
    images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    images[-1] = cv2.resize(images[-1], (512, 64))

images_tensor = tf.stack([tf.convert_to_tensor(image, dtype=tf.float32) for image in images])

custom_objects = {
    'ResidualBlock': ComplexResidualBlock  # Define the custom layer in the dictionary
}


model = keras.models.load_model("model.keras", custom_objects=custom_objects)

predictions = model.predict(images_tensor)


predictions_string = [prediction > .5 if "Right" else "Left" for prediction in predictions]

print(predictions_string)