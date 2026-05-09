# -*- coding: utf-8 -*-
"""
Created on Sat May  9 09:30:27 2026

@author: Avinaba Mukherjee
"""

import streamlit as st

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np
from PIL import Image

# Load model
model = load_model("resnet_cancer_model.h5")

# Title
st.title("Histopathologic Cancer Detection")

# Upload image
uploaded_file = st.file_uploader(
    "Upload Histopathology Image",
    type=["tif","png","jpg","jpeg"]
)

# Prediction function
def predict_image(img):

    img = img.resize((224,224))

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    return prediction[0][0]

# If image uploaded
if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image")

    prediction = predict_image(img)

    if prediction > 0.5:
        st.error(f"Cancer Detected ({prediction:.4f})")
    else:
        st.success(f"No Cancer Detected ({prediction:.4f})")
        
        