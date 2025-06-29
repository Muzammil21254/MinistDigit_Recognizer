import streamlit as st
import numpy as np
from PIL import Image
import cv2
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Title
st.title("🧠 MNIST Digit Recognizer")
st.markdown("Upload a 28x28 digit image to classify it.")

# Load and train the model once
@st.cache_resource
def train_model():
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X / 255.0
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

model = train_model()

# Upload an image
uploaded_file = st.file_uploader("Upload a 28x28 grayscale digit image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")  # grayscale
    img_resized = img.resize((28, 28))
    st.image(img_resized, caption="Uploaded Digit", use_column_width=False)

    # Preprocess image
    img_array = np.array(img_resized)
    img_array = 255 - img_array  # invert colors (white bg, black digit)
    img_flat = img_array.reshape(1, -1) / 255.0  # normalize

    # Predict
    prediction = model.predict(img_flat)[0]
    st.success(f"🎯 Predicted Digit: **{prediction}**")
