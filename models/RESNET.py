# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:49:31 2024

@author: Manodeep
"""

import tensorflow as tf
from tensorflow.keras import layers, models

def resnet_block(input_layer, filters, kernel_size, strides=(1, 1), conv_shortcut=False):
    shortcut = input_layer
    if conv_shortcut:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if conv_shortcut:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides)(input_layer)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def ResNet18(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # Initial convolution layer
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Residual blocks
    x = resnet_block(x, filters=64, kernel_size=(3, 3))
    x = resnet_block(x, filters=64, kernel_size=(3, 3))

    x = resnet_block(x, filters=128, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = resnet_block(x, filters=128, kernel_size=(3, 3))

    x = resnet_block(x, filters=256, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = resnet_block(x, filters=256, kernel_size=(3, 3))

    x = resnet_block(x, filters=512, kernel_size=(3, 3), strides=(2, 2), conv_shortcut=True)
    x = resnet_block(x, filters=512, kernel_size=(3, 3))

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

# Example usage:
input_shape = (224, 224, 3)  # Input shape for the ResNet-18 model
num_classes = 1000  # Number of output classes

# Create ResNet-18 model
resnet18_model = ResNet18(input_shape, num_classes)

# Summary of the model architecture
resnet18_model.summary()




def ResNet34(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # Initial convolution layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128, strides=2, conv_shortcut=True)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)

    x = resnet_block(x, 256, strides=2, conv_shortcut=True)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)

    x = resnet_block(x, 512, strides=2, conv_shortcut=True)
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

# Example usage:
input_shape = (224, 224, 3)  # Input shape for the ResNet-34 model
num_classes = 1000  # Number of output classes

# Create ResNet-34 model
resnet34_model = ResNet34(input_shape, num_classes)

# Summary of the model architecture
resnet34_model.summary()




def ResNet50(input_shape, num_classes):
    input_layer = layers.Input(shape=input_shape)

    # Initial convolution layer
    x = layers.Conv2D(64, 7, strides=2, padding='same')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    # Residual blocks
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)
    x = resnet_block(x, 64)

    x = resnet_block(x, 128, strides=2, conv_shortcut=True)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)
    x = resnet_block(x, 128)

    x = resnet_block(x, 256, strides=2, conv_shortcut=True)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)
    x = resnet_block(x, 256)

    x = resnet_block(x, 512, strides=2, conv_shortcut=True)
    x = resnet_block(x, 512)
    x = resnet_block(x, 512)

    # Global average pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=input_layer, outputs=x)
    return model

# Example usage:
input_shape = (224, 224, 3)  # Input shape for the ResNet-50 model
num_classes = 1000  # Number of output classes

# Create ResNet-50 model
resnet50_model = ResNet50(input_shape, num_classes)

# Summary of the model architecture
resnet50_model.summary()