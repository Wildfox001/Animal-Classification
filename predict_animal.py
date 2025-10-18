# predict_animal.py
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("cattle_buffalo_model.h5")

# Path to the image you want to test
img_path = "dataset/test/cattle/sample.jpg"  # replace with your image path

# Load and preprocess the image
img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # rescale
img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

# Make prediction
prediction = model.predict(img_array)[0][0]

# Interpret prediction
if prediction < 0.5:
    print(f"Predicted: Cattle ({prediction:.2f})")
else:
    print(f"Predicted: Buffalo ({prediction:.2f})")
