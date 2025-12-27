import streamlit as st
import requests

st.title("🌿 Plant Disease Detection")
st.write("Upload une image de feuille, et le modèle prédit la maladie.")

API_URL = "http://127.0.0.1:8000/predict"

uploaded = st.file_uploader("Choisir une image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    st.image(uploaded, caption="Image uploadée", use_column_width=True)

    if st.button("Predict"):
        files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
        r = requests.post(API_URL, files=files)

        if r.status_code == 200:
            data = r.json()
            st.success(f"✅ Prediction: {data['predicted_class']}")
            st.info(f"Confidence: {data['confidence']:.3f}")
        else:
            st.error(f"Erreur API: {r.status_code} - {r.text}")
