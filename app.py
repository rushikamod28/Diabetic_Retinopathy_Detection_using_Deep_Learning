import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model once
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dr_final.h5")
    return model

model = load_model()

# Class names (adjust if needed)
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR"
]

st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader("Upload a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for model
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0  # normalize to [0,1]
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    # Prediction button
    if st.button("Predict"):
        preds = model.predict(img_array)
        pred_class = np.argmax(preds, axis=1)[0]
        confidence = preds[0][pred_class]

        st.write(f"**Prediction:** {class_names[pred_class]}")
        st.write(f"**Confidence:** {confidence*100:.2f}%")
