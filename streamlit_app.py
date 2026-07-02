import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gtts import gTTS
import tempfile, os
from PIL import Image

# -----------------------------
# Load trained model
# -----------------------------
model = load_model("soil_type_model_final.h5")

# -----------------------------
# Class labels
# -----------------------------
class_labels = ["Alluvial soil", "Black soil", "Clay soil", "Red soil"]

# -----------------------------
# Farmer-friendly advice
# -----------------------------
farmer_advice = {
    "Alluvial soil": {
        "crops": "Rice, Wheat, Sugarcane, Maize, Pulses",
        "fertilizers": "Urea, DAP, Organic compost",
        "organic_pesticides": "Neem oil spray, Garlic-chili extract, Cow urine solution",
        "tips": "Needs regular irrigation and timely fertilizer application.",
        "tips_hindi": "नियमित सिंचाई और समय पर उर्वरक का उपयोग आवश्यक है।"
    },
    "Black soil": {
        "crops": "Cotton, Soybean, Groundnut, Wheat, Sorghum",
        "fertilizers": "NPK fertilizers, Farmyard manure, Potash for cotton",
        "organic_pesticides": "Neem cake powder, Tobacco decoction, Trichoderma bio-control",
        "tips": "Holds moisture well but avoid waterlogging.",
        "tips_hindi": "मिट्टी नमी को अच्छी तरह से रखती है लेकिन जलभराव से बचें।"
    },
    "Clay soil": {
        "crops": "Rice, Jute, Wheat",
        "fertilizers": "Compost, Urea, Superphosphate, Green manure",
        "organic_pesticides": "Neem spray, Soap water spray for aphids, Lemongrass oil",
        "tips": "Poor drainage — add organic matter to improve aeration.",
        "tips_hindi": "ड्रेनेज खराब है — वायु संचलन सुधारने के लिए जैविक पदार्थ डालें।"
    },
    "Red soil": {
        "crops": "Groundnut, Millet, Potato, Cotton",
        "fertilizers": "Organic compost, NPK mix, Gypsum to improve fertility",
        "organic_pesticides": "Neem seed kernel extract, Buttermilk spray, Garlic-ginger extract",
        "tips": "Low in nutrients — add compost/manure regularly.",
        "tips_hindi": "पोषक तत्वों में कम — नियमित रूप से कम्पोस्ट/गोबर डालें।"
    }
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Soil Type Predictor", layout="centered")
st.title("🌱 Smart Soil Type Prediction System")
st.write("Upload a soil image to predict soil type and get farming advice with audio.")

# -----------------------------
# Confidence Threshold (adjustable)
# -----------------------------
THRESHOLD = st.slider("Set confidence threshold for soil detection", 0.5, 0.95, 0.75, 0.01)

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("📤 Upload Soil Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Soil Image", use_column_width=True)

    # Save temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        # Preprocess image
        img = Image.open("temp.jpg").convert("RGB")
        img = img.resize((224, 224))
        img_array = np.expand_dims(np.array(img)/255.0, axis=0)

        # Prediction
        predictions = model.predict(img_array)[0]
        top_conf = np.max(predictions)
        predicted_class = class_labels[np.argmax(predictions)]

        # Show model confidence
        st.write(f"🔹 Model confidence for top class: {top_conf:.2f}")

        # -----------------------------
        # Reject non-soil images
        # -----------------------------
        if top_conf < THRESHOLD:
            st.error("❌ Not a soil image (or low confidence). Please upload a valid soil image.")

            # Language selection for error message (kept as before)
            language = st.selectbox("Select Language", ["English", "Hindi"], key="lang_error")
            if language == "Hindi":
                error_text = "❌ यह मिट्टी की छवि नहीं है (विश्वास स्तर बहुत कम है)। कृपया एक मान्य मिट्टी की छवि अपलोड करें।"
                lang_code = "hi"
            else:
                error_text = "❌ This is not a soil image (low confidence). Please upload a valid soil image."
                lang_code = "en"

            st.markdown(error_text)

            # Audio for error message (kept as before, not deleting file)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf:
                temp_file = tf.name
            tts = gTTS(text=error_text, lang=lang_code, slow=False)
            tts.save(temp_file)
            st.audio(temp_file, format="audio/mp3")
            # do not remove the temp file here

        else:
            # -----------------------------
            # Top-3 predictions
            # -----------------------------
            st.subheader("🔎 Top-3 Soil Predictions:")
            top3_idx = predictions.argsort()[-3:][::-1]
            for i, idx in enumerate(top3_idx):
                soil_name = class_labels[idx]
                confidence = predictions[idx]*100
                st.markdown(f"{i+1}. **{soil_name}** — {confidence:.1f}% confidence")

            advice = farmer_advice[predicted_class]

            # -----------------------------
            # Display advice
            # -----------------------------
            st.subheader(f"🧾 Predicted Soil Type: **{predicted_class}**")
            st.markdown(f"### ✅ Recommended Crops:\n{advice['crops']}")
            st.markdown(f"### 💊 Fertilizer Suggestions:\n{advice['fertilizers']}")
            st.markdown(f"🐛 **Organic Pesticides:** {advice['organic_pesticides']}")

            # -----------------------------
            # Language selection for tips & audio (keep as-is for text/tips)
            # -----------------------------
            language = st.selectbox("Select Language", ["English", "Hindi"])

            if language == "Hindi":
                st.markdown(f"💡 **Tips (Hindi):** {advice['tips_hindi']}")
            else:
                st.markdown(f"💡 **Tips (English):** {advice['tips']}")

            # -----------------------------
            # Audio playback (FOR VALID SOIL) - play BOTH English and Hindi audio
            # -----------------------------
            # prepare English audio text
            audio_text_en = f"Predicted Soil Type: {predicted_class}. Recommended crops: {advice['crops']}. Fertilizer suggestions: {advice['fertilizers']}. Organic pesticides: {advice['organic_pesticides']}. Extra tips: {advice['tips']}."
            # prepare Hindi audio text
            audio_text_hi = f"अनुमानित मिट्टी प्रकार: {predicted_class}. सिफारिश की गई फसलें: {advice['crops']}. उर्वरक सुझाव: {advice['fertilizers']}. जैविक कीटनाशक: {advice['organic_pesticides']}. अतिरिक्त सुझाव: {advice['tips_hindi']}."

            # English audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf_en:
                temp_en = tf_en.name
            tts_en = gTTS(text=audio_text_en, lang="en", slow=False)
            tts_en.save(temp_en)
            st.audio(temp_en, format="audio/mp3")

            # Hindi audio
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tf_hi:
                temp_hi = tf_hi.name
            tts_hi = gTTS(text=audio_text_hi, lang="hi", slow=False)
            tts_hi.save(temp_hi)
            st.audio(temp_hi, format="audio/mp3")

            # Note: files are not deleted immediately so Streamlit can play them reliably.
            # You can remove them later if desired.

    except Exception as e:
        st.error(f"❌ Error: Not a valid image file. ({str(e)})")
