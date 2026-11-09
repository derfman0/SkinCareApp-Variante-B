"""SkinCareApp - minimalistische Hautanalyse (Variante 2: Hugging Face)."""

import io

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------- Einstellungen ----------
MODEL_ID = "varun1505/face-characteristics"
MAX_FILE_MB = 5
CONFIDENCE_THRESHOLD = 0.50  # darunter: "nicht sicher"
DISCLAIMER = (
    "SkinCareApp is an educational skincare application. "
    "It does not provide medical diagnoses or replace professional medical advice."
)

# Die 7 echten Klassen des Modells -> Namen in der App
LABEL_MAP = {
    "acne": "Skin Imperfections",
    "skin redness": "Redness",
    "skin pigmentation": "Pigmentation",
    "skin texture": "Skin Texture",
    "eyes dark circles": "Dark Circles",
    "puffy eyes": "Puffy Eyes",
    "wrinkled face": "Wrinkles",
}
NOT_SUPPORTED = ["Visible Pores", "Dry Skin"]

st.set_page_config(page_title="SkinCareApp", page_icon="🌸", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 720px;}
    h1, h2, h3 {color: #4A3328; font-weight: 600; letter-spacing: -0.01em;}
    .card {background: #FFFFFF; border-radius: 18px; padding: 1.1rem 1.3rem;
           box-shadow: 0 2px 12px rgba(74,51,40,0.07); margin-bottom: 1rem;}
    .small {color: #8a7468; font-size: 0.85rem;}
    .stButton > button {border-radius: 999px; border: none; background: #D8A7A0;
           color: white; padding: 0.5rem 1.5rem; font-weight: 500;}
    .stButton > button:hover {background: #c8938b; color: white;}
    .stProgress > div > div > div > div {background-color: #D8A7A0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- Modell ----------
@st.cache_resource(show_spinner="Loading AI model (first start can take a minute)...")
def load_classifier():
    """Lädt das Hugging-Face-Modell einmalig und merkt es sich."""
    from transformers import pipeline

    return pipeline("image-classification", model=MODEL_ID, top_k=None)


def classify(image):
    """Gibt eine Liste (App-Name, Original-Label, Score) zurück, sortiert nach Score."""
    classifier = load_classifier()
    results = []
    for item in classifier(image):
        original = item["label"]
        key = original.replace("_", " ").replace("-", " ").strip().lower()
        results.append((LABEL_MAP.get(key, original), original, float(item["score"])))
    return sorted(results, key=lambda r: r[2], reverse=True)


# ---------- Bild-Hilfsfunktionen ----------
def load_valid_image(file):
    """Prüft Größe und Format. Gibt (Bild, Fehlertext) zurück."""
    data = file.getvalue()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"The file is larger than {MAX_FILE_MB} MB. Please choose a smaller photo."
    try:
        Image.open(io.BytesIO(data)).verify()  # prüft, ob die Datei ein echtes Bild ist
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "This file could not be read as an image. Please try another photo."


def reset_image():
    """Entfernt das Bild (neue Widget-Keys leeren Upload und Kamera)."""
    st.session_state.reset_count = st.session_state.get("reset_count", 0) + 1
    st.session_state.result = None


# ---------- Seiten ----------
def page_analysis():
    st.subheader("Skin Analysis")
    st.markdown(
        "<p class='small'>Your photo is processed inside this app, is not stored and is not "
        "sent to any external AI service.</p>",
        unsafe_allow_html=True,
    )

    n = st.session_state.get("reset_count", 0)
    source = st.radio("Image source", ["Upload from Gallery", "Take a Selfie"],
                      horizontal=True, label_visibility="collapsed")

    # Nur EINE Quelle ist aktiv, damit nie zwei Bilder gleichzeitig analysiert werden.
    if source == "Upload from Gallery":
        file = st.file_uploader("Upload a selfie (JPG, JPEG, PNG)",
                                type=["jpg", "jpeg", "png"], key=f"upload_{n}")
    else:
        st.caption("Your browser will ask for camera permission. "
                   "If the camera does not work, use the upload instead.")
        file = st.camera_input("Take a Selfie", key=f"camera_{n}")

    if file is None:
        return

    image, error = load_valid_image(file)
    if error:
        st.error(error)
        return

    st.image(image)
    col1, col2 = st.columns(2)
    col1.button("Remove Image", on_click=reset_image)
    analyze = col2.button("Analyze My Skin")

    if analyze:
        try:
            with st.spinner("Analyzing your skin..."):
                st.session_state.result = classify(image)
        except Exception:  # absichtlich allgemein: keine Bilddaten in Fehlermeldungen
            st.session_state.result = None
            st.error("The AI model could not be loaded or run. Please try again later.")

    if st.session_state.get("result"):
        show_results(st.session_state.result)


def show_results(results):
    st.subheader("Results")
    top_name, _, top_score = results[0]

    if top_score < CONFIDENCE_THRESHOLD:
        st.warning("We couldn't confidently analyze this image. Please try another photo.")
        st.markdown(
            f"<p class='small'>Category: Unclear / No Detection "
            f"(highest model confidence {top_score:.0%}, threshold {CONFIDENCE_THRESHOLD:.0%}).</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='card'><b>Most likely visible feature:</b> {top_name}<br>"
            f"<span class='small'>Model confidence: {top_score:.0%}. The model picks "
            f"<u>one</u> category per image; it does not detect several features at once.</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("**All model confidences**")
    for name, _, score in results:
        st.progress(min(max(score, 0.0), 1.0), text=f"{name}: {score:.0%}")

    st.markdown(
        f"<p class='small'>Not detected by this model: {', '.join(NOT_SUPPORTED)}. "
        "Confidence values are the model's own output, not a medical measure.</p>",
        unsafe_allow_html=True,
    )


def page_guide():
    st.subheader("Skincare Guide")
    st.caption("General information only. No medical advice, no guaranteed results.")
    topics = {
        "Skin Imperfections": "Cleanse gently twice a day with a mild cleanser. Avoid squeezing or "
                              "scrubbing, as this can irritate the skin. Choose non-comedogenic products.",
        "Redness": "Use fragrance-free, mild products. Avoid known irritants such as harsh scrubs and "
                   "very hot water. Keep your routine simple. Persistent redness: ask a doctor.",
        "Pigmentation": "Daily sun protection (SPF 30+) is the most important habit, as sun can darken "
                        "spots. Be patient and consistent. New or changing spots: see a doctor.",
        "Skin Texture": "Gentle cleansing and regular moisturizer help skin feel smooth. Avoid "
                        "over-exfoliating, which can irritate the skin barrier.",
        "Dark Circles": "Enough sleep, sun protection and a gentle moisturizer for the eye area can help. "
                        "Dark circles are often genetic.",
        "Puffy Eyes": "Sleep, hydration and less salty food can help. A cool compress may feel "
                      "refreshing. Persistent swelling: ask a doctor.",
        "Wrinkles": "Daily sunscreen and moisturizer are the basics for healthy-looking skin. "
                    "Wrinkles are a natural part of aging.",
        "Visible Pores (not detected by the AI)": "Pore size is mostly genetic. Gentle cleansing and "
                                                  "a light moisturizer keep skin looking even.",
        "Dry Skin (not detected by the AI)": "Moisturize regularly, ideally on slightly damp skin. Use a "
                                             "mild cleanser and lukewarm water to support the skin barrier.",
    }
    for title, text in topics.items():
        with st.expander(title):
            st.write(text)


def page_routine():
    st.subheader("My Skincare Routine")
    morning = [("Cleanse", "Wash your face gently with a mild cleanser."),
               ("Moisturize", "Apply a moisturizer suited to your skin."),
               ("Sun Protection", "Finish with sunscreen, SPF 30 or higher.")]
    evening = [("Cleanse", "Clean your face and remove sunscreen."),
               ("Optional Care", "Add a mild, gentle product if you like."),
               ("Moisturize", "Apply moisturizer.")]
    for heading, steps in (("☀️ Morning", morning), ("🌙 Evening", evening)):
        st.markdown(f"**{heading}**")
        for i, (title, text) in enumerate(steps, 1):
            st.markdown(
                f"<div class='card'><b>Step {i} - {title}</b><br>"
                f"<span class='small'>{text}</span></div>",
                unsafe_allow_html=True,
            )


def page_about():
    st.subheader("About & Privacy")
    st.write("SkinCareApp is a school project. It uses a pretrained AI model from Hugging Face "
             f"(`{MODEL_ID}`) to classify a selfie into one visible skin-feature category.")
    st.markdown("**Limits of the model**")
    st.write("It assigns one category per image, reports a self-stated accuracy of about 57 %, and "
             "does not detect Dry Skin or Visible Pores. Results can be wrong.")
    st.markdown("**Privacy**")
    st.write("Photos are processed in memory only, are not saved and are not sent to external AI "
             "services. Use 'Remove Image' to delete the current photo from the app.")
    st.info(DISCLAIMER)


# ---------- Start ----------
st.title("🌸 SkinCareApp")
st.caption("Gentle, AI-assisted insights into visible skin features.")

tab_analysis, tab_guide, tab_routine, tab_about = st.tabs(
    ["Analysis", "Skincare Guide", "Routine", "About & Privacy"])
with tab_analysis:
    page_analysis()
with tab_guide:
    page_guide()
with tab_routine:
    page_routine()
with tab_about:
    page_about()

st.markdown(f"<p class='small'>{DISCLAIMER}</p>", unsafe_allow_html=True)
