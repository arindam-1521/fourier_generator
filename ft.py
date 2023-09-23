import streamlit as st
import cv2
import numpy as np
import os
import time

# Create a temporary directory if it doesn't exist
if not os.path.exists("temp"):
    os.makedirs("temp")

# Streamlit UI
st.title("Fourier Transform Image App")

# File uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Generate a unique filename based on the current timestamp
    file_extension = uploaded_file.name.split(".")[-1]
    unique_filename = f"{int(time.time())}.{file_extension}"

    # Save the uploaded file to a temporary directory
    with open(os.path.join("temp", unique_filename), "wb") as f:
        f.write(uploaded_file.read())

    # Load the input image
    image = cv2.imread(os.path.join("temp", unique_filename), cv2.IMREAD_GRAYSCALE)

    # Normalize image data to [0.0, 1.0] range
    image_normalized = image.astype(np.float32) / 255.0

    # Perform Fourier Transform
    f_transform = np.fft.fft2(image_normalized)
    f_transform_shifted = np.fft.fftshift(f_transform)

    # Calculate magnitude spectrum (logarithmic scale for visualization)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

    # Perform Inverse Fourier Transform
    f_transform_inverse_shifted = np.fft.ifftshift(f_transform_shifted)
    image_back = np.fft.ifft2(f_transform_inverse_shifted)
    image_back = np.abs(image_back)

    # Manually adjust pixel values to [0.0, 1.0]
    magnitude_spectrum = (magnitude_spectrum - np.min(magnitude_spectrum)) / (np.max(magnitude_spectrum) - np.min(magnitude_spectrum))
    image_back = (image_back - np.min(image_back)) / (np.max(image_back) - np.min(image_back))

    # Display the original, Fourier transform, and inverse Fourier transform images
    st.subheader("Input Image")
    st.image(image_normalized, use_column_width=True, channels="GRAY")

    st.subheader("Fourier Transform Image")
    st.image(magnitude_spectrum, use_column_width=True, channels="GRAY")

    st.subheader("Inverse Fourier Transform Image")
    st.image(image_back, use_column_width=True, channels="GRAY")
