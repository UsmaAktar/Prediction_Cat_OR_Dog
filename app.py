
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from keras.models import load_model

# -------------------------------
# App title
# -------------------------------
st.title("🐱🐶 Cats vs Dogs Classifier")
st.write("Upload an image and the model will predict whether it's a **Cat** or a **Dog**.")

# -------------------------------
# Image upload
# -------------------------------
uploaded_file = st.file_uploader(
    "Upload a cat or dog image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------
# Prediction
# -------------------------------
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]

    # Output
    st.subheader("Prediction Result:")
    if prediction > 0.5:
        st.success(f"🐶 Dog ({prediction:.2f})")
    else:
        st.success(f"🐱 Cat ({1 - prediction:.2f})")

    # Confidence bar
    st.progress(float(prediction))
