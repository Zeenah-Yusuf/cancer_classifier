# model_training.py

import os
import numpy as np
import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import joblib

# Define paths
train_dir = r"C:\Users\zeena\Desktop\cancer_classifier\cancer_classification_data\train"
val_dir = r"C:\Users\zeena\Desktop\cancer_classifier\cancer_classification_data\val"
test_dir = r"C:\Users\zeena\Desktop\cancer_classifier\cancer_classification_data\test"

for path in [train_dir, val_dir, test_dir]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing folder: {path}")

# Image size and batch
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Rescale pixel values
train_gen = ImageDataGenerator(rescale=1./255)
val_gen = ImageDataGenerator(rescale=1./255)
test_gen = ImageDataGenerator(rescale=1./255)

# Load images from folders
train_data = train_gen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data = val_gen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_data = test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Save label encoder
joblib.dump(train_data.class_indices, 'label_encoder.pkl')

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=val_data, epochs=10)

# Save model
model.save('cancer_classifier_model.h5')
