import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('model.h5')

# Function to predict class (normal or positive) for a single image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Scale pixel values to [0, 1]
    prediction = model.predict(img_array)
    if prediction < 0.5:
        return "Normal"
    else:
        return "Positive"

# Example usage:
image_path = "/home/ftpb19/Shubham/Thesis_Project/Kidney_Cancer/Normal/kidney_normal_0025.jpg"
#image_path = "/home/ftpb19/Shubham/Thesis_Project/Kidney_Cancer/Tumor/kidney_tumor_5000.jpg"
prediction = predict_image(image_path)
print("Prediction Result:", prediction)