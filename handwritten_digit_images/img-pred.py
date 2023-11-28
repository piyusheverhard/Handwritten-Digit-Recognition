import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os


def adjust_brightness_contrast(img):
    # Read the image
    alpha = 2
    beta = 50

    # Apply the brightness and contrast adjustment
    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    return adjusted_img

def preprocess_image(image_path, custom_threshold_value):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Show the original image
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.show()
    
    # Show the adjusted image
    img_adjusted = adjust_brightness_contrast(img)
    plt.imshow(img_adjusted, cmap='gray')
    plt.title('Adjusted Image')
    plt.show()

    # Resize the image to 28x28
    img_resized = cv2.resize(img_adjusted, (28, 28), interpolation=cv2.INTER_AREA)

    # Invert the resized image
    img_resized_inverted = (img_resized)

    # Show the resized image
    plt.imshow(img_resized_inverted, cmap='gray')
    plt.title('Resized and Inverted Image')
    plt.show()

    # Apply threshold to convert it into black and white
    _, img_threshold = cv2.threshold(img_resized_inverted, custom_threshold_value, 255, cv2.THRESH_BINARY)

    # Show the thresholded image
    plt.imshow(img_threshold, cmap='gray')
    plt.title('Thresholded Image')
    plt.show()

    # Convert the image to a matrix of 0 and 1
    img_matrix = img_threshold / 255

    return img_matrix

# Load the saved model
model = load_model('mnist_cnn_model.h5')

# Replace 'path_to_image' with the path to your image
for i in range(10):
    folder_path = './captured_images/' + str(i)
    file_list = os.listdir(folder_path)
    correct = 0
    total = len(file_list)
    for file_name in file_list:
        image_path = folder_path + '/' + file_name
        custom_threshold_value = 120
        custom_image_matrix = preprocess_image(image_path, custom_threshold_value)
        custom_image_matrix = np.expand_dims(custom_image_matrix, axis=0)  # Add batch dimension
        print(custom_image_matrix)

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
        if (predicted_digit == i):
            correct += 1
        print("accuracy: ", correct / total)

