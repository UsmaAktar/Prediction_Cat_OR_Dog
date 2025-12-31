
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
