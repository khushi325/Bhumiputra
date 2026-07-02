import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("soil_type_model_final.h5")

# Class labels (must match your training folders)
class_labels = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]

# Farmer-friendly advice dictionary with fertilizers
farmer_advice = {
    "Alluvial soil": {
        "crops": "Rice, Wheat, Sugarcane, Maize, Pulses",
        "fertilizers": "Urea, DAP (Diammonium Phosphate), Organic compost",
        "tips": "Needs regular irrigation and timely fertilizer application."
    },
    "Black soil": {
        "crops": "Cotton, Soybean, Groundnut, Wheat, Sorghum",
        "fertilizers": "NPK fertilizers, Farmyard manure, Potash for cotton",
        "tips": "Holds moisture well but avoid waterlogging."
    },
    "Clay soil": {
        "crops": "Rice, Jute, Wheat",
        "fertilizers": "Compost, Urea, Superphosphate, Green manure",
        "tips": "Poor drainage — add organic matter to improve aeration."
    },
    "Red soil": {
        "crops": "Groundnut, Millet, Potato, Cotton",
        "fertilizers": "Organic compost, NPK mix, Gypsum to improve fertility",
        "tips": "Low in nutrients — add compost/manure regularly."
    }
}

# Streamlit UI
st.set_page_config(page_title="Soil Type Predictor", layout="centered")
st.title("🌱 Soil Type Prediction System")
st.write("Upload a soil image to predict its type and get easy farming advice.")

uploaded_file = st.file_uploader("📤 Upload Soil Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Soil Image", use_column_width=True)

    # Save temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess image (resize to 224x224 to match model input)
    img = image.load_img("temp.jpg", target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_labels[np.argmax(predictions)]

    # Show result
    st.subheader(f"🧾 Predicted Soil Type: **{predicted_class}**")

    advice = farmer_advice[predicted_class]

    st.markdown(f"""
    ### ✅ Recommended Crops:
    {advice['crops']}

    ### 💊 Fertilizer Suggestions:
    {advice['fertilizers']}

    ### 💡 Extra Tips:
    {advice['tips']}
    """)
