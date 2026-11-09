"""SkinCareApp - minimalistische Hautanalyse (Variante 2: Hugging Face)."""

import io
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ---------- Einstellungen ----------
MODEL_ID = "varun1505/face-characteristics"
MAX_FILE_MB = 5
CONFIDENCE_THRESHOLD = 0.50  # darunter: "nicht sicher"
DISCLAIMER = (
    "Die SkinCareApp ist eine Lern-App zur Hautpflege. "
    "Sie stellt keine medizinische Diagnose und ersetzt keine ärztliche Beratung."
)

# Die 7 echten Klassen des Modells -> deutsche Namen
LABEL_MAP = {
    "acne": "Hautunreinheiten",
    "skin redness": "Rötungen",
    "skin pigmentation": "Pigmentierung",
    "skin texture": "Hautstruktur",
    "eyes dark circles": "Augenringe",
    "puffy eyes": "Geschwollene Augen",
    "wrinkled face": "Falten",
}
NOT_SUPPORTED = ["Poren", "Trockene Haut"]

# Erklärung und Pflegetipps zu jedem Merkmal (allgemeine Informationen)
CARE = {
    "Hautunreinheiten": (
        "Sichtbare Unreinheiten wie Pickel oder Mitesser.",
        ["Zweimal täglich sanft reinigen, nicht schrubben.",
         "Nicht ausdrücken, das kann Reizungen und Narben fördern.",
         "Produkte wählen, die als 'nicht komedogen' gekennzeichnet sind.",
         "Bei starken oder schmerzhaften Unreinheiten: Hautarzt fragen."]),
    "Rötungen": (
        "Sichtbare Rötungen der Haut.",
        ["Milde, parfümfreie Produkte verwenden.",
         "Sehr heißes Wasser und grobe Peelings meiden.",
         "Die Pflegeroutine einfach halten.",
         "Bei anhaltender Rötung: Hautarzt fragen."]),
    "Pigmentierung": (
        "Sichtbare dunklere Flecken auf der Haut.",
        ["Täglich Sonnenschutz (mindestens LSF 30) verwenden.",
         "Geduld haben: Hautveränderungen brauchen Zeit.",
         "Neue oder sich verändernde Flecken immer ärztlich anschauen lassen."]),
    "Hautstruktur": (
        "Eine auffällige Oberflächenstruktur der Haut.",
        ["Sanft reinigen und regelmäßig eincremen.",
         "Nicht zu oft peelen, das reizt die Hautbarriere.",
         "Täglich Sonnenschutz verwenden."]),
    "Augenringe": (
        "Dunkle Schatten unter den Augen. Sie sind oft erblich bedingt.",
        ["Ausreichend schlafen.",
         "Sonnenschutz und eine milde Augenpflege verwenden.",
         "Ein kühles Tuch kann sich erfrischend anfühlen."]),
    "Geschwollene Augen": (
        "Sichtbare Schwellungen im Augenbereich.",
        ["Genug trinken und ausreichend schlafen.",
         "Ein kühles Tuch kann helfen.",
         "Bei dauerhaften Schwellungen: Arzt fragen."]),
    "Falten": (
        "Sichtbare Linien und Falten. Sie gehören zum natürlichen Altern.",
        ["Täglich Sonnenschutz ist die wichtigste Basis.",
         "Regelmäßig Feuchtigkeitspflege verwenden.",
         "Kein Produkt kann Falten garantiert entfernen."]),
}

# Einfache Stichwortliste für den Inhaltsstoffe-Check
INGREDIENTS = {
    "parfum": ("⚠️", "Duftstoff: kann empfindliche Haut reizen."),
    "fragrance": ("⚠️", "Duftstoff: kann empfindliche Haut reizen."),
    "limonene": ("⚠️", "Duftstoff-Allergen: kann bei empfindlicher Haut reizen."),
    "linalool": ("⚠️", "Duftstoff-Allergen: kann bei empfindlicher Haut reizen."),
    "alcohol denat": ("⚠️", "Vergällter Alkohol: kann austrocknen und reizen."),
    "sodium lauryl sulfate": ("⚠️", "Starkes Reinigungsmittel: kann die Haut austrocknen."),
    "menthol": ("⚠️", "Kann bei empfindlicher Haut reizen."),
    "retinol": ("ℹ️", "Starker Wirkstoff: langsam einführen, Sonnenschutz nutzen, "
                      "in der Schwangerschaft ärztlich klären."),
    "salicylic acid": ("ℹ️", "Säure (BHA): kann bei Unreinheiten genutzt werden, "
                             "aber auch reizen. Vorsichtig dosieren."),
    "glycolic acid": ("ℹ️", "Säure (AHA): macht die Haut sonnenempfindlicher. Sonnenschutz nutzen."),
    "niacinamide": ("✅", "Wird meist gut vertragen."),
    "glycerin": ("✅", "Feuchtigkeitsspendend."),
    "hyaluronic": ("✅", "Feuchtigkeitsspendend."),
    "ceramide": ("✅", "Unterstützt die Hautbarriere."),
    "panthenol": ("✅", "Beruhigend und feuchtigkeitsspendend."),
}

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
@st.cache_resource(show_spinner="KI-Modell wird geladen (der erste Start kann etwas dauern) ...")
def load_model():
    """Lädt Bildprozessor und Modell einmalig und merkt sie sich."""
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify(image):
    """Gibt eine Liste (deutscher Name, Original-Label, Score) zurück, sortiert nach Score."""
    import torch

    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

    results = []
    for index, score in enumerate(probabilities.tolist()):
        original = model.config.id2label[index]
        key = original.replace("_", " ").replace("-", " ").strip().lower()
        results.append((LABEL_MAP.get(key, original), original, float(score)))
    return sorted(results, key=lambda r: r[2], reverse=True)


# ---------- Bild-Hilfsfunktionen ----------
def load_valid_image(file):
    """Prüft Größe und Format. Gibt (Bild, Fehlertext) zurück."""
    data = file.getvalue()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"Die Datei ist größer als {MAX_FILE_MB} MB. Bitte wähle ein kleineres Foto."
    try:
        Image.open(io.BytesIO(data)).verify()  # prüft, ob es ein echtes Bild ist
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image).convert("RGB")
        return image, None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "Diese Datei konnte nicht als Bild gelesen werden. Bitte versuche ein anderes Foto."


def check_photo_quality(image):
    """Grobe Prüfung von Helligkeit und Schärfe. Gibt eine Liste von Hinweisen zurück."""
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=float)
    brightness = gray.mean()
    # Schärfe: Varianz der Kantenstärke (Laplace-Filter)
    laplace = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
    sharpness = laplace.var()

    hints = []
    if brightness < 70:
        hints.append("Das Foto ist ziemlich dunkel. Nimm es bei Tageslicht auf.")
    if brightness > 200:
        hints.append("Das Foto ist sehr hell. Vermeide direktes, grelles Licht.")
    if sharpness < 30:
        hints.append("Das Foto wirkt unscharf. Halte die Kamera ruhig.")
    return hints


def reset_image():
    """Entfernt das Bild (neue Widget-Keys leeren Upload und Kamera)."""
    st.session_state.reset_count = st.session_state.get("reset_count", 0) + 1
    st.session_state.result = None


def build_report(results):
    """Erstellt einen Textbericht (ohne Foto)."""
    top_name, _, top_score = results[0]
    lines = [
        "SkinCareApp - Analysebericht",
        datetime.now().strftime("Datum: %d.%m.%Y %H:%M"),
        "",
        f"Wahrscheinlichstes Merkmal: {top_name} ({top_score:.0%})",
        "",
        "Alle Modellwerte:",
    ]
    lines += [f"- {name}: {score:.0%}" for name, _, score in results]
    lines += ["", f"Modell: {MODEL_ID}", DISCLAIMER]
    return "\n".join(lines)


# ---------- Seiten ----------
def page_analysis():
    st.subheader("Hautanalyse")
    st.markdown(
        "<p class='small'>Dein Foto wird nur in dieser App verarbeitet, nicht gespeichert "
        "und nicht an einen externen KI-Dienst gesendet.</p>",
        unsafe_allow_html=True,
    )

    n = st.session_state.get("reset_count", 0)
    source = st.radio("Bildquelle", ["Foto hochladen", "Selfie aufnehmen"],
                      horizontal=True, label_visibility="collapsed")

    # Nur EINE Quelle ist aktiv, damit nie zwei Bilder gleichzeitig analysiert werden.
    if source == "Foto hochladen":
        file = st.file_uploader("Selfie hochladen (JPG, JPEG, PNG)",
                                type=["jpg", "jpeg", "png"], key=f"upload_{n}")
    else:
        st.caption("Dein Browser fragt nach der Erlaubnis für die Kamera. "
                   "Falls die Kamera nicht funktioniert, nutze den Foto-Upload.")
        file = st.camera_input("Selfie aufnehmen", key=f"camera_{n}")

    if file is None:
        return

    image, error = load_valid_image(file)
    if error:
        st.error(error)
        return

    st.image(image)
    for hint in check_photo_quality(image):
        st.warning(hint)

    col1, col2 = st.columns(2)
    col1.button("Bild entfernen", on_click=reset_image)
    analyze = col2.button("Haut analysieren")

    if analyze:
        try:
            with st.spinner("Deine Haut wird analysiert ..."):
                results = classify(image)
            st.session_state.result = results
            history = st.session_state.get("history", [])
            history.append({"Zeit": datetime.now().strftime("%H:%M:%S"),
                            "Ergebnis": results[0][0],
                            "Sicherheit": f"{results[0][2]:.0%}"})
            st.session_state.history = history
        except Exception as error_info:  # absichtlich allgemein; keine Bilddaten im Text
            st.session_state.result = None
            st.error("Das KI-Modell konnte nicht geladen oder ausgeführt werden. "
                     "Bitte versuche es später noch einmal.")
            with st.expander("Technische Details"):
                st.code(f"{type(error_info).__name__}: {error_info}")

    if st.session_state.get("result"):
        show_results(st.session_state.result)


def show_results(results):
    st.subheader("Ergebnis")
    top_name, _, top_score = results[0]

    if top_score < CONFIDENCE_THRESHOLD:
        st.warning("Wir konnten dieses Bild nicht sicher auswerten. Bitte versuche ein anderes Foto.")
        st.markdown(
            f"<p class='small'>Kategorie: Unklar / keine Erkennung "
            f"(höchster Modellwert {top_score:.0%}, Schwelle {CONFIDENCE_THRESHOLD:.0%}).</p>",
            unsafe_allow_html=True,
        )
    else:
        explanation, tips = CARE.get(top_name, ("", []))
        st.markdown(
            f"<div class='card'><b>Wahrscheinlichstes sichtbares Merkmal:</b> {top_name}<br>"
            f"<span class='small'>{explanation} Sicherheit des Modells: {top_score:.0%}. "
            f"Das Modell wählt <u>eine</u> Kategorie pro Bild und erkennt nicht mehrere "
            f"Merkmale gleichzeitig.</span></div>",
            unsafe_allow_html=True,
        )
        if tips:
            st.markdown("**Allgemeine Pflegetipps**")
            for tip in tips:
                st.markdown(f"- {tip}")

    st.markdown("**Alle Modellwerte**")
    for name, _, score in results:
        st.progress(min(max(score, 0.0), 1.0), text=f"{name}: {score:.0%}")

    st.markdown(
        f"<p class='small'>Von diesem Modell nicht erkannt: {', '.join(NOT_SUPPORTED)}. "
        "Die Werte sind die Ausgabe des Modells und kein medizinisches Maß.</p>",
        unsafe_allow_html=True,
    )
    st.download_button("Bericht herunterladen (ohne Foto)", build_report(results),
                       file_name="skincare_bericht.txt")


def page_history():
    st.subheader("Verlauf")
    st.caption("Nur für diese Sitzung. Es werden keine Fotos gespeichert.")
    history = st.session_state.get("history", [])
    if not history:
        st.info("Noch keine Analyse in dieser Sitzung.")
        return
    st.table(history)
    if st.button("Verlauf löschen"):
        st.session_state.history = []
        st.rerun()


def page_quiz():
    st.subheader("Hauttyp-Quiz")
    st.caption("Zwei Fragen für allgemeine Hinweise. Keine Diagnose.")

    sun = st.radio(
        "Wie reagiert deine Haut auf Sonne ohne Schutz?",
        ["Verbrennt schnell, bräunt kaum",
         "Verbrennt oft, bräunt langsam",
         "Verbrennt manchmal, bräunt gut",
         "Verbrennt selten, bräunt schnell"],
        index=None,
    )
    feel = st.radio(
        "Wie fühlt sich deine Haut mittags an?",
        ["Spannt oder fühlt sich trocken an",
         "Normal und angenehm",
         "Glänzt, vor allem an Stirn und Nase"],
        index=None,
    )

    if sun is None or feel is None:
        return

    st.markdown("**Deine Hinweise**")
    if sun in ("Verbrennt schnell, bräunt kaum", "Verbrennt oft, bräunt langsam"):
        st.write("☀️ Deine Haut ist sonnenempfindlich: Nutze täglich Sonnenschutz mit LSF 50, "
                 "meide die Mittagssonne und creme bei Bedarf nach.")
    else:
        st.write("☀️ Auch bei weniger sonnenempfindlicher Haut gilt: Täglich Sonnenschutz "
                 "mit mindestens LSF 30 schützt vor Hautschäden.")

    if feel.startswith("Spannt"):
        st.write("💧 Trockene Tendenz: Milde Reinigung, lauwarmes Wasser und eine reichhaltigere "
                 "Feuchtigkeitscreme können helfen.")
    elif feel.startswith("Normal"):
        st.write("💧 Normale Haut: Sanft reinigen, eincremen und Sonnenschutz genügen meist.")
    else:
        st.write("💧 Ölige Tendenz: Leichte, nicht komedogene Pflege (z. B. Gel-Creme) passt oft gut. "
                 "Trotzdem eincremen, auch fettige Haut braucht Feuchtigkeit.")
    st.markdown(f"<p class='small'>{DISCLAIMER}</p>", unsafe_allow_html=True)


def page_ingredients():
    st.subheader("Inhaltsstoffe-Check")
    st.caption("Füge die Zutatenliste (INCI) eines Pflegeprodukts ein.")
    text = st.text_area("Zutatenliste", height=150,
                        placeholder="Aqua, Glycerin, Parfum, Alcohol Denat., Niacinamide ...")
    if not text.strip():
        return

    lowered = text.lower()
    found = []
    shown = set()
    for keyword, (icon, note) in INGREDIENTS.items():
        if keyword in lowered and (icon, note) not in shown:
            shown.add((icon, note))
            found.append((keyword, icon, note))

    if not found:
        st.info("Keine Stoffe aus unserer kleinen Liste gefunden. Das heißt nicht, "
                "dass das Produkt für dich passend ist.")
    for keyword, icon, note in found:
        st.markdown(f"{icon} **{keyword.title()}**: {note}")
    st.markdown("<p class='small'>⚠️ möglicher Reizstoff · ℹ️ Wirkstoff mit Hinweisen · ✅ pflegend. "
                "Das ist nur eine einfache Stichwortliste und keine Bewertung der Verträglichkeit.</p>",
                unsafe_allow_html=True)


def page_about():
    st.subheader("Info & Datenschutz")
    st.write("Die SkinCareApp ist ein Schulprojekt. Sie nutzt ein vortrainiertes KI-Modell von "
             f"Hugging Face (`{MODEL_ID}`), das ein Selfie einer sichtbaren Hautmerkmal-Kategorie zuordnet.")
    st.markdown("**Grenzen des Modells**")
    st.write("Es wählt nur eine Kategorie pro Bild, gibt eine selbst angegebene Genauigkeit von "
             "etwa 57 % an und erkennt keine Poren und keine trockene Haut. Ergebnisse können falsch sein.")
    st.markdown("**Datenschutz**")
    st.write("Fotos werden nur im Arbeitsspeicher verarbeitet, nicht gespeichert und nicht an externe "
             "KI-Dienste gesendet. Mit „Bild entfernen“ löschst du das aktuelle Foto aus der App.")
    st.info(DISCLAIMER)


# ---------- Start ----------
st.title("🌸 SkinCareApp")
st.caption("Sanfte, KI-gestützte Einblicke in sichtbare Hautmerkmale.")

tabs = st.tabs(["Analyse", "Verlauf", "Hauttyp-Quiz", "Inhaltsstoffe", "Info"])
with tabs[0]:
    page_analysis()
with tabs[1]:
    page_history()
with tabs[2]:
    page_quiz()
with tabs[3]:
    page_ingredients()
with tabs[4]:
    page_about()

st.markdown(f"<p class='small'>{DISCLAIMER}</p>", unsafe_allow_html=True)
