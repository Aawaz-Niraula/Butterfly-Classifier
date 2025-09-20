import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import matplotlib.pyplot as plt

# -------------------------
# Settings (match your training)
# -------------------------
img_height, img_width = 224, 224

# -------------------------
# Load trained model
# -------------------------
@st.cache_resource
def load_model_cache():
    return load_model("butterfly_mobilenetv2.h5")

model = load_model_cache()

# -------------------------
# Load class names
# -------------------------
with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

# -------------------------
# App title
# -------------------------
st.title("Butterfly Species Classifier 🦋")
st.write("Upload an image of a butterfly and the model will predict its species specially if its prats or shrssss 😋😋😋. Guys Alik dhang ko pictures hai")

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Choose a butterfly image...", type=["jpg","jpeg","png","webp"])

if uploaded_file is not None:
    # Open image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = img.resize((img_width, img_height))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # MobileNetV2 preprocessing BEFORE prediction

    # Predict
    prediction = model.predict(img_array)[0]

    # Top 3 predictions
    top3_idx = prediction.argsort()[-3:][::-1]
    top3_labels = [class_names[i] for i in top3_idx]
    top3_conf = [prediction[i]*100 for i in top3_idx]

    # Display results
    st.subheader("Top Predictions:")
    for label, conf in zip(top3_labels, top3_conf):
        st.write(f"{label}: {conf:.2f}%")

    # -------------------------
    # Bar chart of top-3 predictions
    # -------------------------
    fig, ax = plt.subplots()
    ax.barh(top3_labels[::-1], top3_conf[::-1], color='skyblue')
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Top-3 Predictions")
    st.pyplot(fig)
