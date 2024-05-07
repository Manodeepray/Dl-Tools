# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:52:49 2024

@author: MANODEEP
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def res_block(x, filters, kernel_size=(3, 3), strides=(1, 1), padding='same'):
    shortcut = x

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding=padding)(x)
    x = layers.BatchNormalization()(x)

    if strides != (1, 1) or x.shape[-1] != shortcut.shape[-1]:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding=padding)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def ResUNet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Contracting Path (Encoder)
    conv1 = res_block(inputs, 64)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = res_block(pool1, 128)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = res_block(pool2, 256)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = res_block(pool3, 512)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    # Bottom
    conv5 = res_block(pool4, 1024)

    # Expanding Path (Decoder)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    concat6 = layers.concatenate([up6, conv4])
    conv6 = res_block(concat6, 512)

    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    concat7 = layers.concatenate([up7, conv3])
    conv7 = res_block(concat7, 256)

    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    concat8 = layers.concatenate([up8, conv2])
    conv8 = res_block(concat8, 128)

    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    concat9 = layers.concatenate([up9, conv1])
    conv9 = res_block(concat9, 64)

    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), activation='softmax')(conv9)

    model = models.Model(inputs, outputs)
    return model

# Example usage:
input_shape = (256, 256, 3)  # Input shape for the ResUNet model
num_classes = 2  # Number of output classes

# Create ResUNet model
resunet_model = ResUNet(input_shape, num_classes)

# Summary of the model architecture
resunet_model.summary()
