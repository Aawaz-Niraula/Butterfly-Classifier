import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------
# Load model & class names
# -----------------------
model = load_model("butterfly_mobilenetv2.h5")
print("Model input shape:", model.input_shape)

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

print("Number of classes:", len(class_names))
print("First 10 class names:", class_names[:10])

# -----------------------
# Prepare test samples
# Replace with 5–10 known training images + their labels
# -----------------------
# Example: you must update paths + true labels
sample_known = [
    (r"C:\Users\Aawazn\Downloads\archive (1)\train\Image_1.jpg", "SOUTHERN DOGFACE"),
    (r"C:\Users\Aawazn\Downloads\archive (1)\train\Image_2.jpg", "ADONIS"),
    (r"C:\Users\Aawazn\Downloads\archive (1)\train\Image_3.jpg", "BROWN SIPROETA"),
    (r"train/Image_4.jpg", "MONARCH"),
    (r"C:\Users\Aawazn\Downloads\archive (1)\train\Image_5.jpg", "GREEN CELLED CATTLEHEART"),
]

_, IMG_H, IMG_W, _ = model.input_shape

def load_and_prep(img_path, mode="mobilenet"):
    """Load and preprocess an image for prediction."""
    img = Image.open(img_path).convert("RGB").resize((IMG_W, IMG_H))
    arr = np.array(img).astype(np.float32)
    if mode == "mobilenet":
        arr = preprocess_input(arr)  # MobileNetV2 style: [-1,1]
    else:
        arr = arr / 255.0            # plain scaling
    return np.expand_dims(arr, axis=0)

# -----------------------
# Test both preprocessing styles
# -----------------------
for mode in ["mobilenet", "divide255"]:
    print("\n=== Testing with", mode, "===")
    correct = 0
    for path, true_label in sample_known:
        x = load_and_prep(path, mode=mode)
        pred = model.predict(x, verbose=0)[0]
        pred_idx = pred.argmax()
        pred_label = class_names[pred_idx]
        if pred_label == true_label:
            correct += 1
        else:
            print("WRONG:", path, "->", pred_label, "(expected:", true_label, ")")
    print("Sample accuracy with", mode, ":", correct, "/", len(sample_known))
