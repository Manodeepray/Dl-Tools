# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:44:48 2024

@author: MANODEEP
"""


import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense

# Define the YOLOv8 architecture
def YOLOv8(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Define your YOLOv8 architecture here
    # Example layers:
    x = Conv2D(64, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = LeakyReLU(alpha=0.1)(x)
    outputs = Dense(num_classes + 5)(x)

    model = Model(inputs, outputs)
    return model

# Example usage:
# Define input shape and number of classes
input_shape = (416, 416, 3)
num_classes = 10

# Create YOLOv8 model
yolov8_model = YOLOv8(input_shape, num_classes)

# Compile the model
yolov8_model.compile(optimizer='adam', loss='mse')

# Train the model with your dataset
# Replace 'X_train' and 'y_train' with your training data
# yolov8_model.fit(X_train, y_train, epochs=10, batch_size=32)

# After training, you can use the model for inference
# Replace 'image' with your input image
# output = yolov8_model.predict(image)

