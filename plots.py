# -*- coding: utf-8 -*-
"""
Created on Wed May  8 00:54:47 2024

@author: KIIT
"""

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

def compute_f1_score(history):
    # Compute F1 score from precision and recall values in the history object
    precision = history.history['precision']
    recall = history.history['recall']
    f1 = [2 * (p * r) / (p + r) if p + r != 0 else 0 for p, r in zip(precision, recall)]
    return f1

def plot_f1_score(history):
    f1 = compute_f1_score(history)
    plt.plot(f1, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Over Epochs')
    plt.legend()
    plt.show()

# Assuming you have already trained your model and have a history object
# history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=num_epochs, batch_size=batch_size)

# Plot the F1 score graph
plot_f1_score(history)


import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# Assuming you have already trained your model and have a history object
# history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=num_epochs, batch_size=batch_size)

# Plot the loss graph
plot_loss(history)
import matplotlib.pyplot as plt

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.show()

# Assuming you have already trained your model and have a history object
# history = model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=num_epochs, batch_size=batch_size)

# Plot the accuracy graph
plot_accuracy(history)
