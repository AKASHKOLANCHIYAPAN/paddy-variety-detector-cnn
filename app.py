import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image

# Load model
model = load_model("paddy_model.keras")
class_names = ['banidhan07', 'banidhan08', 'banidhan10', 'banidhan11', 'basmathi']  # change to your actual varieties

st.title("Paddy Variety Detector 🌾")
st.write("Upload a paddy image to classify its variety.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((128,128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Variety: {predicted_class}")
