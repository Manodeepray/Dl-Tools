# -*- coding: utf-8 -*-
"""
Created on Wed May  8 01:08:22 2024

@author: manodeep
"""

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define your model
model = tf.keras.Sequential([
    # Add your layers here
])

# Compile your model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint_callback = ModelCheckpoint(filepath='checkpoint.h5', save_best_only=True, monitor='val_loss', mode='min')
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train your model with callbacks
history = model.fit(train_data, train_labels, 
                    validation_data=(val_data, val_labels), 
                    epochs=num_epochs, 
                    batch_size=batch_size, 
                    callbacks=[checkpoint_callback, early_stopping_callback])

# After training, load the best model
best_model = tf.keras.models.load_model('checkpoint.h5')
