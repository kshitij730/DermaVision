import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os

# Load models with caching
@st.cache_resource  # Cache the models to avoid reloading
def load_hair_model():
    return tf.keras.models.load_model('VGG19-Final.h5')

@st.cache_resource
def load_skin_model():
    return tf.keras.models.load_model('skin_cancer_detection_model.h5')

hair_model = load_hair_model()
skin_model = load_skin_model()

# Define class labels
hair_diseases = ['Alopecia Areata', 'Contact Dermatitis', 'Folliculitis', 'Head Lice', 'Lichen Planus', 
                 'Male Pattern Baldness', 'Psoriasis', 'Seborrheic Dermatitis', 'Telogen Effluvium', 'Tinea Capitis']

skin_diseases = ['Actinic Keratoses', 'Basal Cell Carcinoma', 'Benign Keratosis', 'Dermatofibroma', 'Melanoma', 'Nevus', 'Vascular Lesion']

# Cure dictionary for hair diseases
def predict_cure(disease_name):
    cures = {
        'Alopecia Areata': 'Cure: Steroid injections or topical treatments.',
        'Contact Dermatitis': 'Cure: Avoid allergens, use corticosteroid creams.',
        'Folliculitis': 'Cure: Antibiotics or antifungal creams.',
        'Head Lice': 'Cure: Over-the-counter lice treatments.',
        'Lichen Planus': 'Cure: Topical corticosteroids or oral treatments.',
        'Male Pattern Baldness': 'Cure: Minoxidil, Finasteride, or hair transplant.',
        'Psoriasis': 'Cure: Topical ointments, phototherapy, or systemic treatments.',
        'Seborrheic Dermatitis': 'Cure: Antifungal treatments or corticosteroid creams.',
        'Telogen Effluvium': 'Cure: Stress management, balanced diet, and vitamins.',
        'Tinea Capitis': 'Cure: Oral antifungal medications.'
    }
    return cures.get(disease_name, 'Cure information not available.')

# Function to preprocess image for both models
def preprocess_image(img):
    img = img.resize((224, 224))  # Resize image to match model input size
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to make predictions
def predict_hair_disease(img_array):
    prediction = hair_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return hair_diseases[predicted_class]

def predict_skin_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = skin_model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return skin_diseases[predicted_class], confidence

# Streamlit app
st.title('Skin & Hair Disease Prediction App')

# Selection for disease type
disease_type = st.radio("Select Disease Type:", ('Hair Disease', 'Skin Cancer'))

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    if disease_type == 'Hair Disease':
        img_array = preprocess_image(img)
        disease_name = predict_hair_disease(img_array)
        st.write(f"Predicted Disease: {disease_name}")
        st.write(predict_cure(disease_name))
    
    else:  # Skin Cancer
        with open("temp_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        predicted_class, confidence = predict_skin_disease("temp_image.jpg")
        st.write(f"### Predicted Class: {predicted_class}")
        st.write(f"### Confidence: {confidence * 100:.2f}%")