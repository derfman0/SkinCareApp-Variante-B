import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ---------------------------------------------------------
# SkinCareApp – Hugging Face edition
# Das Modell wird automatisch von Hugging Face geladen.
# Es wird KEINE Modelldatei im GitHub-Repository gespeichert.
# ---------------------------------------------------------

MODEL_ID = "LaurianeMD/vit-skin-disease"
MAX_UPLOAD_MB = 10
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="SkinCareApp",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dezentes, Streamlit-freundliches Design
st.markdown(
    """
    <style>
    .stApp { background: #FBF6F1; color: #4B302B; }
    h1, h2, h3 { color: #4B302B; }
    [data-testid="stSidebar"] { background: #F4E8E1; }
    .hero {
        background: #F7E7E2; padding: 2rem; border-radius: 18px;
        border: 1px solid #EBD4CD; margin-bottom: 1rem;
    }
    .soft-card {
        background: #FFFFFF; padding: 1.1rem; border-radius: 15px;
        border: 1px solid #EADBD4; min-height: 125px;
    }
    .note {
        background: #F7E7E2; padding: 1rem 1.2rem; border-radius: 12px;
        border: 1px solid #EBD4CD; margin-top: 1rem;
    }
    div.stButton > button[kind="primary"] {
        background: #C87979; color: white; border: 0; border-radius: 12px;
    }
    div.stButton > button[kind="primary"]:hover {
        background: #AE6266; color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Das Laden ist gecacht: Nach dem ersten Download wird das Modell
# innerhalb der laufenden App nicht bei jedem Klick neu geladen.
@st.cache_resource(show_spinner=False)
def load_model():
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def predict(image: Image.Image):
    processor, model = load_model()
    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.inference_mode():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    confidence, class_index = torch.max(probabilities, dim=0)
    label = model.config.id2label[int(class_index.item())]
    return label, float(confidence.item())


def show_home():
    st.markdown(
        """
        <div class="hero">
          <h1>SkinCareApp</h1>
          <h3>Discover your skin.</h3>
          <p>Explore an AI-powered image classification demo and find
          simple, general skincare information in one place.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("A simple space for skincare")
        st.write(
            "Upload a photo or take a selfie, then run the pretrained "
            "Hugging Face image-classification model."
        )
        if st.button("Start Skin Analysis →", type="primary", use_container_width=True):
            st.session_state.page = "Skin Analysis"
            st.rerun()
    with right:
        st.markdown(
            """
            <div class="soft-card">
              <h3>🌸 Your skincare space</h3>
              <p>AI image classification</p>
              <p>General skincare tips</p>
              <p>Morning & evening routine</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="note"><b>Important:</b> This is an educational project, '
        'not a medical tool. It does not diagnose skin conditions.</div>',
        unsafe_allow_html=True,
    )


def show_analysis():
    st.title("Skin Analysis")
    st.write("Choose one image source. The selected image is processed for this analysis.")

    source = st.radio(
        "Choose image source",
        ["Upload from gallery", "Take a selfie"],
        horizontal=True,
    )

    image_file = None
    if source == "Upload from gallery":
        image_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG image",
            type=["jpg", "jpeg", "png"],
        )
    else:
        image_file = st.camera_input("Take a selfie")

    if image_file is None:
        st.info("Upload an image or take a selfie to continue.")
        return

    if image_file.size > MAX_UPLOAD_MB * 1024 * 1024:
        st.error(f"Please choose an image smaller than {MAX_UPLOAD_MB} MB.")
        return

    try:
        image = Image.open(image_file)
        image.verify()
        image_file.seek(0)
        image = Image.open(image_file).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("This file could not be opened as a valid image. Please try another one.")
        return

    st.subheader("Image preview")
    st.image(image, use_container_width=True)

    if st.button("Analyze My Image", type="primary", use_container_width=True):
        # Vorherige Ergebnisse entfernen, damit kein altes Ergebnis stehen bleibt.
        st.session_state.pop("prediction", None)
        with st.spinner("Loading the model and analyzing the image..."):
            try:
                label, confidence = predict(image)
                st.session_state.prediction = {
                    "label": label,
                    "confidence": confidence,
                }
            except Exception as exc:
                st.error(
                    "The model could not analyze this image. "
                    "Check your internet connection and try again."
                )
                st.caption(f"Technical detail: {type(exc).__name__}: {exc}")

    result = st.session_state.get("prediction")
    if result:
        st.divider()
        st.subheader("Analysis result")
        if result["confidence"] < CONFIDENCE_THRESHOLD:
            st.warning(
                "The model's confidence is low. Please do not interpret this "
                "prediction as a reliable result."
            )
        st.markdown(
            f'<div class="soft-card"><h3>{result["label"]}</h3>'
            f'<p>Predicted model class</p></div>',
            unsafe_allow_html=True,
        )
        st.write(f"Model confidence: **{result['confidence'] * 100:.1f}%**")
        st.progress(result["confidence"])
        st.caption(
            "The confidence is the model's softmax score, not the probability "
            "that a medical condition is present."
        )
        st.markdown(
            '<div class="note"><b>Model limitation:</b> This pretrained model '
            'classifies dermatology-related image classes. It was not trained '
            'to independently detect the SkinCareApp categories such as visible '
            'pores, dry skin, redness or pigmentation. The result is not a diagnosis.</div>',
            unsafe_allow_html=True,
        )


def show_guide():
    st.title("Skincare Guide")
    st.write("General information only. Keep routines gentle and avoid products that irritate your skin.")
    tips = {
        "Gentle cleansing": "Use a mild cleanser and avoid scrubbing aggressively.",
        "Moisturizing": "A suitable moisturizer can help support the skin barrier.",
        "Sun protection": "Use broad-spectrum sunscreen during the day, following its instructions.",
        "Sensitive-feeling skin": "Introduce new products gradually and stop using products that cause irritation.",
    }
    for heading, body in tips.items():
        with st.expander(heading):
            st.write(body)


def show_routine():
    st.title("My Skincare Routine")
    morning, evening = st.columns(2)
    with morning:
        st.subheader("☀️ Morning")
        st.markdown("1. Gentle cleanse (if needed)\n\n2. Moisturizer\n\n3. Sunscreen")
    with evening:
        st.subheader("🌙 Evening")
        st.markdown("1. Gentle cleanse\n\n2. Optional gentle care\n\n3. Moisturizer")


def show_about():
    st.title("About & Privacy")
    st.write(
        "SkinCareApp is a school project demonstrating how a pretrained "
        "image-classification model can be connected to a Streamlit application."
    )
    st.subheader("About the AI model")
    st.write(f"Model repository: `{MODEL_ID}`")
    st.write(
        "The model files are downloaded from Hugging Face when the app loads. "
        "They are not included in this GitHub repository."
    )
    st.subheader("Privacy")
    st.write(
        "This code does not intentionally save uploaded photos to a project folder "
        "or database. However, the image is sent to the app's server for inference. "
        "The app uses a model downloaded from Hugging Face; it does not call the "
        "Hugging Face hosted inference API in this implementation."
    )
    st.info(
        "Do not upload sensitive or identifiable photos unless you understand "
        "the hosting and data-processing arrangements of the deployed app."
    )
    st.caption(
        "SkinCareApp is an educational skincare application. It does not provide "
        "medical diagnoses or replace professional medical advice."
    )


# Navigation
if "page" not in st.session_state:
    st.session_state.page = "Home"

st.sidebar.title("🌸 SkinCareApp")
page_names = ["Home", "Skin Analysis", "Skincare Guide", "My Routine", "About & Privacy"]
selected = st.sidebar.radio(
    "Navigation",
    page_names,
    index=page_names.index(st.session_state.page),
)
st.session_state.page = selected

if selected == "Home":
    show_home()
elif selected == "Skin Analysis":
    show_analysis()
elif selected == "Skincare Guide":
    show_guide()
elif selected == "My Routine":
    show_routine()
else:
    show_about()
