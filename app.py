import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

MODEL_REPO = "SopanhaZinII/facial-condition-model"
MODEL_FILE = "finetuned_best.h5"
CLASS_NAMES = ["Acne", "Dry Skin", "Oily Skin", "Dark Spots", "Wrinkles"]

st.set_page_config(page_title="SkinCareApp", page_icon="🧴")
st.title("🧴 SkinCareApp")
st.subheader("Experimentelle Hautbild-Analyse mit KI")
st.write("Lade ein Foto hoch und erhalte eine experimentelle Einschätzung zu fünf Hautmerkmalen.")
st.warning("Schulprojekt, keine medizinische Diagnose. Ergebnisse können falsch sein und ersetzen keine dermatologische Beratung. Lade möglichst keine Fotos anderer Personen hoch.")

@st.cache_resource(show_spinner="KI-Modell wird von Hugging Face geladen …")
def load_model():
    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return tf.keras.models.load_model(path, compile=False)

def prepare_image(image):
    image = image.convert("RGB").resize((224, 224))
    # Die Modellkarte nennt 224x224x3, dokumentiert die Normalisierung aber nicht eindeutig.
    return np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

uploaded = st.file_uploader("Hautfoto hochladen", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Hochgeladenes Bild", use_container_width=True)
    if st.button("Hautbild analysieren", type="primary"):
        try:
            model = load_model()
            prediction = np.asarray(model.predict(prepare_image(image), verbose=0)).reshape(-1)
            if len(prediction) != len(CLASS_NAMES):
                st.error(f"Unerwartete Modellausgabe: {len(prediction)} Werte statt 5.")
            else:
                results = sorted(zip(CLASS_NAMES, prediction.tolist()), key=lambda x: x[1], reverse=True)
                st.success("Analyse abgeschlossen.")
                st.caption("Modell-Scores, keine klinisch validierten Wahrscheinlichkeiten. Klassenreihenfolge und Bildvorverarbeitung müssen noch mit dem Modellautor verifiziert werden.")
                for label, score in results:
                    st.write(f"**{label}**")
                    st.progress(float(np.clip(score, 0, 1)))
                    st.write(f"Score: {score:.3f}")
                tips = {
                    "Acne": "Sanfte Reinigung; Haut nicht ausdrücken. Bei anhaltenden Beschwerden dermatologisch beraten lassen.",
                    "Dry Skin": "Milde Reinigung und unparfümierte Feuchtigkeitspflege können helfen.",
                    "Oily Skin": "Leichte, nicht-komedogene Pflege verwenden; übermäßiges Waschen vermeiden.",
                    "Dark Spots": "Täglicher Sonnenschutz kann helfen, UV-bedingte Verstärkung zu vermeiden.",
                    "Wrinkles": "Sonnenschutz und eine passende Feuchtigkeitspflege unterstützen eine einfache Routine."
                }
                st.markdown("### Allgemeine Pflegetipps")
                for label, score in results:
                    if score >= 0.5:
                        st.markdown(f"- **{label}:** {tips[label]}")
        except Exception as error:
            st.error("Die Analyse konnte nicht durchgeführt werden.")
            st.exception(error)

st.divider()
st.markdown("### Einfache Routine")
a, b = st.columns(2)
with a:
    st.markdown("**Morgens**")
    st.write("1. Milde Reinigung\n2. Feuchtigkeitspflege\n3. Sonnenschutz (LSF 30+)")
with b:
    st.markdown("**Abends**")
    st.write("1. Sonnenschutz sanft entfernen\n2. Milde Reinigung\n3. Feuchtigkeitspflege")
st.caption("Schulprojekt · Keine medizinische Beratung.")
