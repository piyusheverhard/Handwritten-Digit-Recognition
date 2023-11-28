import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt

def adjust_brightness_contrast(img):
    alpha = 2
    beta = 50
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img

def preprocess_custom_image(image_path, custom_threshold_value):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_adjusted = adjust_brightness_contrast(img)
    img_resized = cv2.resize(img_adjusted, (28, 28), interpolation=cv2.INTER_AREA)
    img_resized_inverted = cv2.bitwise_not(img_resized)
    _, img_threshold = cv2.threshold(img_resized_inverted, custom_threshold_value, 255, cv2.THRESH_BINARY)
    img_matrix = img_threshold / 255
    return img_matrix

# Load the saved model
model = load_model('mnist_cnn_model.h5')

# Path to your custom image
custom_image_path = 'path/to/your/custom/image.png'
custom_threshold_value = 20

# Preprocess the custom image
custom_image_matrix = preprocess_custom_image(custom_image_path, custom_threshold_value)
custom_image_matrix = np.expand_dims(custom_image_matrix, axis=0)  # Add batch dimension

# Make predictions
predictions = model.predict(custom_image_matrix)

# Get the predicted digit (class with the highest probability)
predicted_digit = np.argmax(predictions[0])

# Display the predicted digit and its probability/confidence
print(f'The predicted digit is: {predicted_digit}')
print(f'Probability/Confidence: {predictions[0][predicted_digit]:.4f}')

# Display probabilities for all classes
for i, probability in enumerate(predictions[0]):
    print(f'Probability for digit {i}: {probability:.4f}')
