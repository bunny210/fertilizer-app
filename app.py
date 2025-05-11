import streamlit as st
import pandas as pd
import pickle
import base64

def set_bg(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call it before title
set_bg("bg1.jpeg")


# Load model and encoders
model = pickle.load(open('model/fertilizer_model.pkl', 'rb'))
soil_encoder = pickle.load(open('model/soil_encoder.pkl', 'rb'))
crop_encoder = pickle.load(open('model/crop_encoder.pkl', 'rb'))
fertilizer_decoder = pickle.load(open('model/fert_encoder.pkl', 'rb'))

st.markdown(
    "<h1 style='text-align: center; color: white;'> Fertilizer Recommendation System</h1>",
    unsafe_allow_html=True
)


# Input fields
temperature = st.number_input("Temperature (°C)", min_value=0)
humidity = st.number_input("Humidity (%)", min_value=0)
moisture = st.number_input("Soil Moisture", min_value=0)
soil_type = st.selectbox("Soil Type", soil_encoder.classes_)
crop_type = st.selectbox("Crop Type", crop_encoder.classes_)
nitrogen = st.number_input("Nitrogen level", min_value=0)
potassium = st.number_input("Potassium level", min_value=0)
phosphorous = st.number_input("Phosphorous level", min_value=0)

if st.button("Get Fertilizer Recommendation"):
    # Encode inputs
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    input_data = [[
        temperature, humidity, moisture, soil_encoded,
        crop_encoded, nitrogen, potassium, phosphorous
    ]]

    prediction = model.predict(input_data)[0]
    fertilizer = fertilizer_decoder.inverse_transform([prediction])[0]

    st.markdown(f"✅ **Recommended Fertilizer:** <span style='color:white'>{fertilizer}</span>", unsafe_allow_html=True)

