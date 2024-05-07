
import os
import cv2
import numpy as np
import tensorflow as tf
import keras

images = []

for filename in os.listdir(r"updated_images"):
    images.append(cv2.imread(r"updated_images/" + filename)[897:1153 ,101:2149])
    images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    images[-1] = cv2.resize(images[-1], (512, 64))

images_tensor = tf.stack([tf.convert_to_tensor(image, dtype=tf.float32) for image in images])

model = keras.models.load_model("model.keras")

predictions = model.predict(images_tensor)


predictions_string = [prediction > .5 if "Right" else "Left" for prediction in predictions]

print(predictions_string)