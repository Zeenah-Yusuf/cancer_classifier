# evaluate_model.py

import tensorflow as tf
from keras.src.legacy.preprocessing.image import ImageDataGenerator
import json

# Load model
model = tf.keras.models.load_model('cancer_classifier_model.h5')

# Define test data path
test_dir = r"C:\Users\zeena\Desktop\cancer_classifier\cancer_classification_data\test"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Load test data
test_gen = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

# Evaluate
loss, accuracy = model.evaluate(test_data)

# Save results
results = {"Test Accuracy": accuracy, "Test Loss": loss}
with open("evaluation_results.json", "w") as f:
    json.dump(results, f)

