from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2

# Load your pretrained model
model = load_model("paddy_model.keras")

# Example: test with one image
img = cv2.imread("sample_paddy.jpg")  # put a test image in same folder
img = cv2.resize(img, (128,128))      # must match your training size
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)
predicted_class = np.argmax(prediction)

print("Predicted Paddy Variety:", predicted_class)
