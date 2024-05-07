
import os
import cv2
import numpy as np
import tensorflow as tf
import keras
from keras import layers
images = []

class ResidualBlock(layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.shortcut = layers.Conv2D(filters, (1, 1), padding='same')  # Shortcut for residual connection

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # Residual connection
        shortcut = self.shortcut(inputs)
        return layers.add([x, shortcut])
    
    def get_config(self):
        config = super().get_config()  # Get base layer config
        config.update({
            'filters': self.conv1.filters,
            'kernel_size': self.conv1.kernel_size,
        })
        return config


for filename in os.listdir(r"updated_images"):
    images.append(cv2.imread(r"updated_images/" + filename)[897:1153 ,101:2149])
    images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    images[-1] = cv2.resize(images[-1], (512, 64))

images_tensor = tf.stack([tf.convert_to_tensor(image, dtype=tf.float32) for image in images])

custom_objects = {
    'ResidualBlock': ResidualBlock  # Define the custom layer in the dictionary
}


model = keras.models.load_model("model.keras", custom_objects=custom_objects)

predictions = model.predict(images_tensor)


predictions_string = [prediction > .5 if "Right" else "Left" for prediction in predictions]

print(predictions_string)