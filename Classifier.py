#This is the beginning of the end for the project

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore

# #clean the data and put it into a csv

# Load Data

images = []
labels = []
handedness_images = []

count_left = 0
count_right = 0

for filename in os.listdir(r"Survey Items"):
    images.append(cv2.imread(r"Survey Items/" + filename)[1317:1573,101:2149])
    handedness_images.append(cv2.imread(r"Survey Items/" + filename)[910:1000,260:350])
    
    images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
    images[-1] = cv2.resize(images[-1], (512, 64))
    handedness_images[-1] = cv2.cvtColor(handedness_images[-1], cv2.COLOR_BGR2GRAY)



    total_sum_one_image = sum(sum(sublist) for sublist in handedness_images[-1])
    average = total_sum_one_image / (90 * 90)

    writing = images[-1]


    if average > 250: 
        #oversample left images by 5
        count_left += 1
        labels.append(1)
        for i in range(4):
            images.append(cv2.imread(r"Survey Items/" + filename)[1317:1573,101:2149])
            images[-1] = cv2.cvtColor(images[-1], cv2.COLOR_BGR2GRAY)
            images[-1] = cv2.resize(images[-1], (512, 64))
            labels.append(1)
            count_left += 1
    else:
        labels.append(0)
        count_right += 1

#Don't move on if there is a difference in length
print(len(images), len(labels))
assert len(labels) == len(images)

#Convert everything to tensors
labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)
images_tensor = tf.stack([tf.convert_to_tensor(image, dtype=tf.float32) for image in images])


"""-------------------------------------------------Model Building---------------------------------------------------"""

images_np = images_tensor.numpy()  
labels_np = labels_tensor.numpy()  


from sklearn.model_selection import train_test_split

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np, test_size=0.3, random_state=1)

# Further split the training set to get a validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1)



#simple model to test everything
import keras
from keras import layers
# Create a simple CNN for binary classification (left- or right-handed)
# model = keras.Sequential([
#     # Convolutional layer with ReLU activation
#     layers.Conv2D(16, (3, 3), activation='relu', input_shape=(64, 512, 1)),
#     layers.MaxPooling2D((2, 2)),  
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(1, activation='sigmoid')
# ])

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


# Define a more complex model with multiple residual blocks and regularization
model = keras.Sequential([
    # Initial convolutional layers
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(64, 512, 1)),  # First layer with 64 filters
    layers.MaxPooling2D((2, 2)),  # Downsampling

    # Complex residual block with multiple convolutions
    ComplexResidualBlock(64, [(3, 3), (5, 5)]),  # Residual block with variable kernel sizes
    layers.MaxPooling2D((2, 2)),  # Downsampling

    # Another residual block with more filters
    ComplexResidualBlock(128, [(3, 3), (3, 3)]),
    layers.MaxPooling2D((2, 2)),  # Downsampling

    # Adding a third residual block with even more filters
    ComplexResidualBlock(256, [(3, 3), (5, 5)]),  # Increasing complexity with more filters and variable kernels

    layers.Flatten(),  # Flatten the output for dense layers
    layers.Dropout(0.5),  # Dropout to prevent overfitting
    layers.Dense(256, activation='relu'),  # Large dense layer for additional learning capacity
    layers.BatchNormalization(),  # Batch normalization for stability
    layers.Dropout(0.5),  # Additional dropout for regularization
    layers.Dense(128, activation='relu'),  # Another dense layer
    layers.Dense(1, activation='sigmoid')  # Two units for binary classification (with softmax activation)
])


# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=25, batch_size=2, validation_data=(X_val, y_val))

# Save the model for future use
model.save("model.keras")

"""--------------------------------------------------------------Plotting Results---------------------------------------------------------------------------"""

import matplotlib.pyplot as plt
# Evaluate the model with the testing set
test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)



#make the plot of the loss over epochs

# Get loss values over epochs
train_loss = history.history['loss']  # Training loss
val_loss = history.history['val_loss']  # Validation loss

# Get accuracy values over epochs (optional)
train_acc = history.history['accuracy']  # Training accuracy
val_acc = history.history['val_accuracy']  # Validation accuracy


# Create an x-axis for epochs
epochs = range(1, len(train_loss) + 1)

# Plot training and validation loss on the same graph
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_loss, 'b-', label='Training Loss')  # 'b-' for blue line
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')  # 'r-' for red line
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.show()

