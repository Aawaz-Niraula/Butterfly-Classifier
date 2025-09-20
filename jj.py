# Check class name alignment
import pickle
from tensorflow.keras.models import load_model

model = load_model("butterfly_classifier.h5")

with open("class_names.pkl", "rb") as f:
    class_names = pickle.load(f)

print("Number of classes from file:", len(class_names))
print("Model output layer units:", model.output_shape[-1])

print("First 10 class names:", class_names[:10])
