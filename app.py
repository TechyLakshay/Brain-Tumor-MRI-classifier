import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("brain_tumor_model.h5")

# Class names (modify if your classes are different)
class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI scan image to detect the type of brain tumor.")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", use_column_width=True)

    # Preprocess the image
    img = image.convert("RGB").resize((150, 150))  # force 3 channels
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 150, 150, 3)
   

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Display result
    st.markdown(f"### 🧪 Prediction: `{predicted_class}`")
    st.markdown(f"### 🔬 Confidence: `{confidence * 100:.2f}%`")
