"""
SkinCareApp
KI-Hautanalyse, Pflegeplan, Tagebuch, Statistik,
Wirkstoff-Lexikon, Quiz und Inhaltsstoffe-Check.
"""

import io
from datetime import date, datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

# ============================================================
# EINSTELLUNGEN UND DATEN
# ============================================================

MODEL_ID = "varun1505/face-characteristics"
MAX_FILE_MB = 5
MAX_IMAGE_SIZE = 1024  # Bilder werden für die Analyse verkleinert (spart Speicher)
CONFIDENCE_THRESHOLD = 0.50

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

# Merkmal: (Erklärung, Pflegetipps)
CARE = {
    "Hautunreinheiten": ("Sichtbare Unreinheiten wie Pickel oder Mitesser.", [
        "Zweimal täglich sanft reinigen.",
        "Nicht ausdrücken, das kann die Haut reizen.",
        "Nicht komedogene Pflege bevorzugen.",
        "Bei starken oder schmerzhaften Unreinheiten Hautarzt fragen."]),
    "Rötungen": ("Sichtbare Rötungen der Haut.", [
        "Milde und möglichst parfümfreie Produkte verwenden.",
        "Sehr heißes Wasser und grobe Peelings vermeiden.",
        "Die Routine einfach halten.",
        "Bei anhaltender Rötung ärztlich abklären lassen."]),
    "Pigmentierung": ("Sichtbare dunklere Flecken oder ungleichmäßige Pigmentierung.", [
        "Täglich Sonnenschutz verwenden.",
        "Geduldig sein, Hautveränderungen brauchen Zeit.",
        "Neue oder sich verändernde Flecken ärztlich anschauen lassen."]),
    "Hautstruktur": ("Eine auffällige oder ungleichmäßige Oberflächenstruktur.", [
        "Sanft reinigen und regelmäßig Feuchtigkeitspflege verwenden.",
        "Nicht zu häufig peelen.",
        "Täglich Sonnenschutz verwenden."]),
    "Augenringe": ("Dunkle Schatten unter den Augen.", [
        "Ausreichend schlafen.",
        "Eine milde Augenpflege verwenden.",
        "Auch im Augenbereich an Sonnenschutz denken."]),
    "Geschwollene Augen": ("Sichtbare Schwellungen im Augenbereich.", [
        "Ausreichend schlafen und trinken.",
        "Ein kühles Tuch kann angenehm sein.",
        "Bei dauerhaften Schwellungen ärztlich abklären lassen."]),
    "Falten": ("Sichtbare Linien und Falten, die zum natürlichen Altern gehören.", [
        "Täglich Sonnenschutz verwenden.",
        "Regelmäßig Feuchtigkeitspflege verwenden.",
        "Keine Pflege kann Falten garantiert entfernen."]),
}

# Name: (Symbol, Kategorie, Beschreibung, Eigenschaften, Tipp)
ACTIVES = {
    "Niacinamid": ("💧", "Beruhigend",
        "Niacinamid ist eine Form von Vitamin B3 und wird häufig in Pflegeprodukten eingesetzt.",
        ["Unterstützt die Hautbarriere", "Kann Rötungen beruhigen", "Kann das Hautbild ausgleichen"],
        "Interessant bei unreiner oder empfindlicher Haut."),
    "Hyaluronsäure": ("💦", "Feuchtigkeit",
        "Hyaluronsäure bindet Feuchtigkeit und steckt in vielen Feuchtigkeitsprodukten.",
        ["Unterstützt die Feuchtigkeitsversorgung", "Kann die Haut geschmeidiger wirken lassen",
         "Passt zu vielen Hauttypen"],
        "Gut mit einer anschließenden Creme kombinierbar."),
    "Retinol": ("✨", "Anti-Aging",
        "Retinol gehört zu den Vitamin-A-Derivaten und wird in der Hautpflege eingesetzt.",
        ["Unterstützt die Hauterneuerung", "Kann das Hautbild verbessern",
         "Häufig in Anti-Aging-Pflege"],
        "Langsam einführen und tagsüber konsequent Sonnenschutz verwenden."),
    "Salicylsäure": ("🫧", "BHA",
        "Salicylsäure ist eine Beta-Hydroxysäure und wird oft bei unreiner Haut eingesetzt.",
        ["Kann Poren von überschüssigem Talg befreien", "Wird häufig bei Unreinheiten eingesetzt",
         "Kann die Hautoberfläche glätten"],
        "Bei empfindlicher Haut langsam und vorsichtig einsetzen."),
    "Panthenol": ("🌿", "Beruhigend",
        "Panthenol wird häufig in beruhigenden und pflegenden Produkten verwendet.",
        ["Unterstützt die Hautpflege", "Kann beruhigend wirken",
         "Hilft bei trockener oder beanspruchter Haut"],
        "Beliebt in beruhigenden Pflegeprodukten."),
    "Ceramide": ("🛡️", "Hautbarriere",
        "Ceramide sind natürliche Bestandteile der Hautbarriere und kommen auch in Pflegeprodukten vor.",
        ["Unterstützen die Hautbarriere", "Helfen, Feuchtigkeit in der Haut zu halten",
         "Interessant bei trockener Haut"],
        "Eine gute Ergänzung für eine einfache Barriereroutine."),
    "Glycerin": ("💧", "Feuchtigkeit",
        "Glycerin ist ein Feuchthaltemittel und kommt in vielen Cremes und Lotionen vor.",
        ["Bindet Feuchtigkeit", "Unterstützt ein geschmeidiges Hautgefühl",
         "Wird von vielen Hauttypen gut vertragen"],
        "Ein sehr häufiger und vielseitiger Pflegebestandteil."),
}

INGREDIENTS = {
    "parfum": ("⚠️", "Duftstoff: kann empfindliche Haut reizen."),
    "fragrance": ("⚠️", "Duftstoff: kann empfindliche Haut reizen."),
    "limonene": ("⚠️", "Duftstoff-Allergen: kann bei empfindlicher Haut reizen."),
    "linalool": ("⚠️", "Duftstoff-Allergen: kann bei empfindlicher Haut reizen."),
    "alcohol denat": ("⚠️", "Vergällter Alkohol: kann austrocknend oder reizend wirken."),
    "sodium lauryl sulfate": ("⚠️", "Starkes Reinigungsmittel: kann die Haut austrocknen."),
    "menthol": ("⚠️", "Kann bei empfindlicher Haut reizen."),
    "retinol": ("ℹ️", "Starker Wirkstoff: langsam einführen, tagsüber Sonnenschutz verwenden."),
    "salicylic acid": ("ℹ️", "BHA-Säure: hilft oft bei Unreinheiten, kann aber auch reizen."),
    "glycolic acid": ("ℹ️", "AHA-Säure: kann die Haut sonnenempfindlicher machen."),
    "niacinamide": ("✅", "Wird meist gut vertragen."),
    "glycerin": ("✅", "Feuchtigkeitsspendender Inhaltsstoff."),
    "hyaluronic": ("✅", "Feuchtigkeitsspendender Inhaltsstoff."),
    "ceramide": ("✅", "Unterstützt die Hautbarriere."),
    "panthenol": ("✅", "Beruhigend und pflegend."),
}

QUIZ = [
    {"q": "Welcher Schritt gehört morgens besonders zur Basisroutine?",
     "options": ["Sonnenschutz", "Sehr starkes Peeling", "Parfüm auf die Haut",
                 "Gesicht mit heißem Wasser waschen"],
     "answer": "Sonnenschutz",
     "why": "Sonnenschutz ist ein wichtiger Teil einer allgemeinen Morgenroutine."},
    {"q": "Welcher Wirkstoff wird häufig mit Feuchtigkeitspflege verbunden?",
     "options": ["Hyaluronsäure", "Parfum", "Menthol", "Alkohol Denat."],
     "answer": "Hyaluronsäure",
     "why": "Hyaluronsäure wird häufig als feuchtigkeitsbindender Wirkstoff eingesetzt."},
    {"q": "Was ist bei empfindlicher Haut grundsätzlich sinnvoll?",
     "options": ["Milde Pflege verwenden", "Jeden Tag stark peelen",
                 "Sehr heißes Wasser benutzen", "Möglichst viele Produkte gleichzeitig"],
     "answer": "Milde Pflege verwenden",
     "why": "Eine einfache, milde Routine hilft, unnötige Reizung zu vermeiden."},
    {"q": "Welcher Wirkstoff gehört zur Vitamin-A-Gruppe?",
     "options": ["Retinol", "Glycerin", "Hyaluronsäure", "Panthenol"],
     "answer": "Retinol",
     "why": "Retinol gehört zu den Vitamin-A-Derivaten."},
    {"q": "Was sollte man bei einem schmerzhaften Hautproblem tun?",
     "options": ["Bei Bedarf ärztlich abklären lassen", "Immer selbst ausdrücken",
                 "Sehr stark schrubben", "Das Problem ignorieren"],
     "answer": "Bei Bedarf ärztlich abklären lassen",
     "why": "Starke oder schmerzhafte Hautprobleme sollten professionell abgeklärt werden."},
]

# ============================================================
# SEITE EINRICHTEN, DESIGN, HILFSFUNKTIONEN
# ============================================================

st.set_page_config(page_title="SkinCareApp", page_icon="🌸", layout="wide",
                   initial_sidebar_state="expanded")


def html(body, sidebar=False):
    """Zeigt HTML an. Entfernt vorher Einrückungen und Leerzeilen,
    damit Streamlit den Text nicht als Code darstellt."""
    cleaned = "\n".join(line.strip() for line in str(body).splitlines() if line.strip())
    (st.sidebar if sidebar else st).markdown(cleaned, unsafe_allow_html=True)


def space():
    html("<div style='height:1rem'></div>")


def hero(title, subtitle, small=False):
    size = "2rem" if small else "2.6rem"
    html(f"""
    <div class="hero">
    <div class="hero-title" style="font-size:{size};">{title}</div>
    <div class="hero-subtitle">{subtitle}</div>
    </div>
    """)


def card(content):
    html(f'<div class="section-card">{content}</div>')


def empty_state(icon, title, text):
    card(f"""
    <div style="text-align:center; padding:1rem;">
    <div style="font-size:2.8rem;">{icon}</div>
    <div style="font-weight:650; color:#4A3328;">{title}</div>
    <div class="small">{text}</div>
    </div>
    """)


html("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 10% 10%, rgba(232,193,187,0.25), transparent 30%),
        radial-gradient(circle at 90% 15%, rgba(216,167,160,0.18), transparent 28%),
        #FBF8F3;
}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F4E9E1 0%, #FBF8F3 55%, #F7EEE8 100%);
    border-right: 1px solid rgba(130,90,80,0.08);
}
h1, h2, h3, h4 {color: #4A3328 !important; font-weight: 600 !important; letter-spacing: -0.02em;}
p, label {color: #5D463C;}
.small {color: #947C70; font-size: 0.86rem;}
.stButton > button {
    border-radius: 999px; border: 1px solid #D8A7A0; background: #D8A7A0;
    color: white; font-weight: 600; padding: 0.55rem 1.35rem;
}
.stButton > button:hover {background: #C99189; border-color: #C99189; color: white;}
.stButton > button:disabled {background: #E9D6D2; border-color: #E9D6D2; color: #fff;}
.stProgress > div > div > div > div {background-color: #D8A7A0;}
.hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(249,235,230,0.92));
    border: 1px solid rgba(150,105,95,0.12); border-radius: 28px;
    padding: 2rem 2.3rem; box-shadow: 0 15px 45px rgba(93,60,50,0.08); margin-bottom: 1.3rem;
}
.hero-title {font-weight: 650; color: #4A3328; margin-bottom: 0.4rem;}
.hero-subtitle {font-size: 1.05rem; color: #846D62; max-width: 680px; line-height: 1.7;}
.section-card {
    background: rgba(255,255,255,0.85); border: 1px solid rgba(140,100,90,0.10);
    border-radius: 22px; padding: 1.3rem; box-shadow: 0 8px 28px rgba(93,60,50,0.05);
    margin-bottom: 1rem;
}
.feature-card {
    background: rgba(255,255,255,0.9); border: 1px solid rgba(140,100,90,0.10);
    border-radius: 20px; padding: 1.1rem; min-height: 165px;
    box-shadow: 0 8px 24px rgba(93,60,50,0.05); margin-bottom: 1rem;
}
.feature-icon {
    width: 42px; height: 42px; border-radius: 14px; background: #F5E1DB;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; margin-bottom: 0.7rem;
}
.feature-title {color: #4A3328; font-weight: 650; font-size: 1.05rem;}
.feature-text {color: #89736A; font-size: 0.88rem; line-height: 1.5; margin-top: 0.35rem;}
.badge {
    display: inline-block; background: #F4E0DA; color: #7E554C; border-radius: 999px;
    padding: 0.3rem 0.7rem; font-size: 0.75rem; font-weight: 600; margin-right: 0.3rem;
}
.badge-green {background: #E6EFE5; color: #55704F;}
.badge-blue {background: #E7EEF2; color: #566D78;}
.badge-purple {background: #EEE7F2; color: #6E5D78;}
.stat-card {
    background: rgba(255,255,255,0.9); border: 1px solid rgba(140,100,90,0.10);
    border-radius: 20px; padding: 1.1rem; text-align: center;
    box-shadow: 0 7px 22px rgba(93,60,50,0.05);
}
.stat-number {font-size: 2rem; font-weight: 700; color: #4A3328;}
.stat-label {color: #967D72; font-size: 0.8rem; margin-top: 0.15rem;}
.step {
    background: #FFFDFC; border: 1px solid #EEDDD7; border-radius: 16px;
    padding: 0.85rem 1rem; margin-bottom: 0.6rem; color: #4A3328;
}
.step-desc {color: #927A70; font-size: 0.84rem;}
.quote {
    background: linear-gradient(135deg, #F5E5DF, #FAF4EF); border-radius: 20px;
    padding: 1.1rem 1.3rem; color: #76594E; font-style: italic; border-left: 4px solid #D8A7A0;
}
</style>
""")

# ============================================================
# SESSION STATE (Daten gelten nur für den aktuellen Besucher)
# ============================================================

DEFAULTS = {
    "history": [], "result": None, "reset_count": 0,
    "diary": [], "diary_next_id": 1,
    "quiz_index": 0, "quiz_score": 0, "quiz_checked": False, "quiz_correct": False,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================
# KI-MODELL
# ============================================================


@st.cache_resource(show_spinner="KI-Modell wird geladen. Der erste Start kann etwas dauern ...")
def load_model():
    from transformers import AutoModelForImageClassification, ViTImageProcessor

    # Das Modell bringt keine Vorverarbeitungs-Datei mit. Wir nutzen die
    # Standardwerte eines ViT-Modells (224x224 Pixel, Normalisierung mit 0.5).
    processor = ViTImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()
    return processor, model


def classify(image):
    """Gibt eine Liste (deutscher Name, Original-Label, Score) zurück, sortiert nach Score."""
    import torch

    processor, model = load_model()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        probabilities = torch.softmax(model(**inputs).logits, dim=-1)[0]

    results = []
    for index, score in enumerate(probabilities.tolist()):
        original = model.config.id2label[index]
        key = original.replace("_", " ").replace("-", " ").strip().lower()
        results.append((LABEL_MAP.get(key, original), original, float(score)))
    return sorted(results, key=lambda r: r[2], reverse=True)

# ============================================================
# BILD-FUNKTIONEN
# ============================================================


def load_valid_image(file):
    """Prüft Größe und Format. Gibt (Bild, Fehlertext) zurück."""
    data = file.getvalue()
    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return None, f"Die Datei ist größer als {MAX_FILE_MB} MB."
    try:
        Image.open(io.BytesIO(data)).verify()  # prüft, ob es ein echtes Bild ist
        image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
        image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))
        return image, None
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "Diese Datei konnte nicht als Bild gelesen werden."


def check_photo_quality(image):
    """Grobe Prüfung von Helligkeit und Schärfe. Gibt eine Liste von Hinweisen zurück."""
    gray = np.asarray(image.convert("L").resize((256, 256)), dtype=float)
    laplace = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
               + gray[1:-1, :-2] + gray[1:-1, 2:])
    hints = []
    if gray.mean() < 70:
        hints.append("Das Foto ist ziemlich dunkel. Tageslicht wäre besser.")
    if gray.mean() > 200:
        hints.append("Das Foto ist sehr hell. Vermeide grelles Licht.")
    if laplace.var() < 30:
        hints.append("Das Foto wirkt unscharf. Halte die Kamera ruhiger.")
    return hints


def reset_image():
    """Neue Widget-Keys leeren Upload und Kamera."""
    st.session_state.reset_count += 1
    st.session_state.result = None


def build_report(results):
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
    lines += ["", f"Modell: {MODEL_ID}", "", DISCLAIMER]
    return "\n".join(lines)

# ============================================================
# SEITE: STARTSEITE
# ============================================================


def page_home():
    html("""
    <div class="hero">
    <div class="hero-title" style="font-size:2.8rem;">Hallo! 🌸</div>
    <div class="hero-subtitle">
    Willkommen in deiner persönlichen SkinCare-Welt. Analysiere sichtbare
    Hautmerkmale, erstelle deine Routine und beobachte deine Hautentwicklung.
    </div>
    <div style="margin-top:1rem;">
    <span class="badge">KI-Analyse</span>
    <span class="badge badge-green">Pflegeplan</span>
    <span class="badge badge-blue">Tagebuch</span>
    <span class="badge badge-purple">Statistik</span>
    </div>
    </div>
    """)

    st.markdown("### Deine SkinCare-Welt")
    features = [
        ("🤖", "KI-Hautanalyse", "Selfie hochladen und sichtbare Merkmale analysieren."),
        ("🧴", "Pflegeplan", "Deine Morgen- und Abendroutine zum Abhaken."),
        ("📔", "Tagebuch", "Hautzustand und Veränderungen festhalten."),
        ("📊", "Statistik", "Deine Entwicklung als Diagramm."),
        ("🧪", "Wirkstoff-Lexikon", "Wichtige Wirkstoffe einfach erklärt."),
        ("🎯", "Hautpflege-Quiz", "Teste dein Wissen rund um Hautpflege."),
        ("🔎", "Inhaltsstoffe", "INCI-Liste auf bekannte Stoffe prüfen."),
        ("ℹ️", "Info", "Projekt, KI-Modell und Datenschutz."),
    ]
    columns = st.columns(4)
    for index, (icon, title, text) in enumerate(features):
        with columns[index % 4]:
            html(f"""
            <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-text">{text}</div>
            </div>
            """)

    card("""
    <div style="color:#4A3328; font-size:1.15rem; font-weight:650;">✨ Dein heutiger Fokus</div>
    <div style="color:#927A70; margin-top:0.45rem; line-height:1.6;">
    Eine gute Routine muss nicht kompliziert sein. Kleine, regelmäßige Schritte
    sind wichtiger als viele Produkte.
    </div>
    <div class="quote" style="margin-top:1rem;">
    „Weniger, aber bewusst: das ist die Idee hinter deiner Routine.“
    </div>
    """)
    st.info("Tipp: Starte mit der KI-Hautanalyse oder hake deinen Pflegeplan ab.")

# ============================================================
# SEITE: KI-HAUTANALYSE
# ============================================================


def page_analysis():
    hero("🤖 KI-Hautanalyse",
         "Lade ein Selfie hoch oder nutze deine Kamera. Die KI ordnet das Bild "
         "einem sichtbaren Hautmerkmal zu.")
    card("""
    <span class="badge">KI-Analyse</span>
    <span class="badge badge-green">Privatsphäre</span>
    <p style="margin:0.8rem 0 0 0;">
    Dein Foto wird nur in dieser App verarbeitet, nicht gespeichert und
    nicht an einen externen KI-Dienst gesendet.
    </p>
    """)

    n = st.session_state.reset_count
    source = st.radio("Bildquelle", ["📁 Foto hochladen", "📷 Selfie aufnehmen"],
                      horizontal=True, label_visibility="collapsed")

    # Nur EINE Quelle ist aktiv, damit nie zwei Bilder gleichzeitig analysiert werden.
    if source == "📁 Foto hochladen":
        file = st.file_uploader("Selfie hochladen (JPG, JPEG, PNG)",
                                type=["jpg", "jpeg", "png"], key=f"upload_{n}")
    else:
        st.caption("Dein Browser fragt eventuell nach der Erlaubnis für die Kamera. "
                   "Falls sie nicht funktioniert, nutze den Foto-Upload.")
        file = st.camera_input("Selfie aufnehmen", key=f"camera_{n}")

    if file is None:
        empty_state("📸", "Bereit für deine Analyse?", "Lade ein Foto hoch oder öffne deine Kamera.")
        return

    image, error = load_valid_image(file)
    if error:
        st.error(error)
        return

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Dein Foto")
    with col2:
        card("""
        <div style="font-weight:650; color:#4A3328; font-size:1.1rem;">Foto-Qualität</div>
        <p class="small" style="margin-bottom:0;">Für bessere Ergebnisse sollte dein Gesicht
        gut sichtbar und ausreichend beleuchtet sein.</p>
        """)
        for hint in check_photo_quality(image):
            st.warning(hint)

    col1, col2 = st.columns(2)
    col1.button("Bild entfernen", on_click=reset_image)
    analyze = col2.button("✨ Haut analysieren")

    if analyze:
        try:
            with st.spinner("Deine Haut wird analysiert ..."):
                results = classify(image)
            st.session_state.result = results
            st.session_state.history.append({
                "Zeit": datetime.now().strftime("%H:%M:%S"),
                "Ergebnis": results[0][0],
                "Sicherheit": f"{results[0][2]:.0%}",
            })
        except Exception as error_info:  # keine Bilddaten in Fehlermeldungen
            st.session_state.result = None
            st.error("Das KI-Modell konnte nicht geladen oder ausgeführt werden.")
            with st.expander("Technische Details"):
                st.code(f"{type(error_info).__name__}: {error_info}")

    if st.session_state.result:
        show_results(st.session_state.result)

    if st.session_state.history:
        with st.expander("Verlauf dieser Sitzung (ohne Fotos)"):
            st.table(st.session_state.history)


def show_results(results):
    space()
    hero("Analyse abgeschlossen ✨", "Hier siehst du die wichtigsten Ergebnisse.", small=True)
    top_name, _, top_score = results[0]

    if top_score < CONFIDENCE_THRESHOLD:
        st.warning("Wir konnten dieses Bild nicht sicher auswerten. Bitte versuche ein anderes Foto.")
        st.caption(f"Kategorie: Unklar / keine Erkennung (höchster Modellwert "
                   f"{top_score:.0%}, Schwelle {CONFIDENCE_THRESHOLD:.0%}).")
    else:
        explanation, tips = CARE.get(top_name, ("", []))
        card(f"""
        <span class="badge badge-green">Höchster Modellwert</span>
        <h3 style="margin-top:0.7rem;">{top_name}</h3>
        <p>{explanation}</p>
        <div style="font-size:1.8rem; font-weight:700; color:#4A3328;">{top_score:.0%}</div>
        <div class="small">Modell-Sicherheit</div>
        """)
        if tips:
            st.markdown("### 💡 Allgemeine Pflegetipps")
            for tip in tips:
                html(f'<div class="step">🌿 {tip}</div>')

    st.markdown("### 📊 Alle Modellwerte")
    for name, _, score in results:
        st.progress(min(max(score, 0.0), 1.0), text=f"{name} · {score:.0%}")

    st.caption(f"Das Modell wählt eine Kategorie pro Bild. Von diesem Modell nicht erkannt: "
               f"{', '.join(NOT_SUPPORTED)}. Die Werte sind keine medizinischen Messwerte.")
    st.download_button("📄 Analysebericht herunterladen (ohne Foto)", build_report(results),
                       file_name="skincare_bericht.txt", mime="text/plain")

# ============================================================
# SEITE: PFLEGEPLAN
# ============================================================


def page_routine():
    hero("🧴 Persönlicher Pflegeplan", "Eine einfache Morgen- und Abendroutine zum Abhaken.")

    skin_tips = {
        "Noch nicht sicher": "Beobachte, wie sich deine Haut mittags anfühlt, das zeigt oft den Hauttyp.",
        "Normal": "Eine einfache Basispflege genügt meist.",
        "Trocken": "Eine reichhaltigere Creme und mildes Reinigen mit lauwarmem Wasser helfen oft.",
        "Ölig": "Leichte Gel-Cremes passen oft gut. Trotzdem eincremen.",
        "Mischhaut": "Eine leichte Creme für das ganze Gesicht ist meist ein guter Start.",
        "Empfindlich": "Wenige, milde Produkte. Neues immer einzeln testen.",
    }
    focus_tips = {
        "Ausgeglichene Pflege": "Reinigen, eincremen, Sonnenschutz: mehr braucht es für den Anfang nicht.",
        "Feuchtigkeit": "Eine Creme mit Glycerin oder Hyaluronsäure kann sinnvoll sein.",
        "Unreinheiten": "Nicht komedogene Produkte wählen und nicht an Unreinheiten drücken.",
        "Rötungen": "Milde, parfümfreie Produkte bevorzugen und Reizstoffe meiden.",
        "Hautbarriere": "Auf Produkte mit Ceramiden achten und nicht zu oft peelen.",
        "Sonnenschutz": "Täglich Sonnenschutz, auch bei bewölktem Wetter.",
    }
    skin_type = st.selectbox("Welcher Hauttyp passt am ehesten zu dir?", list(skin_tips))
    focus = st.selectbox("Was möchtest du besonders berücksichtigen?", list(focus_tips))

    space()
    routines = [
        ("morning", '<span class="badge">☀️ MORGEN</span>', "Deine Morgenroutine", [
            ("Sanfte Reinigung", "Entfernt Schweiß und überschüssiges Hautfett."),
            ("Feuchtigkeit", "Eine Creme passend zu deinem Hautgefühl."),
            ("Sonnenschutz", "Der wichtigste Schritt am Morgen.")]),
        ("evening", '<span class="badge badge-purple">🌙 ABEND</span>', "Deine Abendroutine", [
            ("Reinigung", "Entfernt Make-up, Schmutz und Sonnenschutz."),
            ("Wirkstoff (optional)", "Nur, wenn er zu deiner Routine passt."),
            ("Feuchtigkeitspflege", "Unterstützt die Haut über Nacht.")]),
    ]
    for column, (prefix, badge, title, steps) in zip(st.columns(2), routines):
        with column:
            html(f'<div class="section-card">{badge}<h3>{title}</h3></div>')
            done = 0
            for step_title, description in steps:
                if st.checkbox(step_title, key=f"{prefix}_{step_title}"):
                    done += 1
                st.caption(description)
            st.progress(done / len(steps), text=f"{done} von {len(steps)} erledigt")

    space()
    html(f"""
    <div class="quote">
    <b>Dein Fokus:</b> {focus}<br>
    <span style="font-size:0.9rem;">{focus_tips[focus]}</span><br>
    <span style="font-size:0.9rem;">Hauttyp {skin_type}: {skin_tips[skin_type]}</span>
    </div>
    """)
    space()
    st.info("Die Routine ist eine allgemeine Orientierung und keine individuelle Behandlung.")

# ============================================================
# SEITE: TAGEBUCH
# ============================================================


def save_diary_entry(entry):
    entry["id"] = st.session_state.diary_next_id
    st.session_state.diary_next_id += 1
    st.session_state.diary.append(entry)


def delete_diary_entry(entry_id):
    st.session_state.diary = [e for e in st.session_state.diary if e["id"] != entry_id]


def page_diary():
    hero("📔 Pflege-Tagebuch",
         "Halte fest, wie sich deine Haut anfühlt, und entdecke Veränderungen. "
         "Die Einträge gelten nur für diese Sitzung und werden nicht gespeichert.")

    col1, col2 = st.columns(2)
    with col1:
        entry_date = st.date_input("Datum", value=date.today())
        skin_score = st.slider("Wie zufrieden bist du heute mit deiner Haut?", 1, 5, 3)
        dryness = st.slider("Trockenheit", 1, 5, 1)
    with col2:
        redness = st.slider("Rötungen", 1, 5, 1)
        impurities = st.slider("Unreinheiten", 1, 5, 1)
        note = st.text_area("Notiz", placeholder="Wie fühlt sich deine Haut heute an?", height=100)

    if st.button("💾 Tagebucheintrag speichern"):
        save_diary_entry({
            "Datum": str(entry_date), "Hautbewertung": skin_score,
            "Trockenheit": dryness, "Rötungen": redness,
            "Unreinheiten": impurities, "Notiz": note,
        })
        st.success("Dein Tagebucheintrag wurde gespeichert.")

    st.markdown("### Deine Einträge")
    diary = st.session_state.diary
    if not diary:
        st.info("Noch keine Einträge vorhanden.")
        return

    for entry in sorted(diary, key=lambda e: (e["Datum"], e["id"]), reverse=True)[:10]:
        with st.expander(f"📅 {entry['Datum']} · Hautgefühl {entry['Hautbewertung']}/5"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Trockenheit", f"{entry['Trockenheit']}/5")
            c2.metric("Rötungen", f"{entry['Rötungen']}/5")
            c3.metric("Unreinheiten", f"{entry['Unreinheiten']}/5")
            if entry["Notiz"]:
                st.write(entry["Notiz"])
            st.button("Eintrag löschen", key=f"delete_{entry['id']}",
                      on_click=delete_diary_entry, args=(entry["id"],))

# ============================================================
# SEITE: STATISTIK
# ============================================================


def stat_card(number, label):
    html(f"""
    <div class="stat-card">
    <div class="stat-number">{number}</div>
    <div class="stat-label">{label}</div>
    </div>
    """)


def page_statistics():
    hero("📊 Haut-Statistik", "Dein Überblick über die Werte aus deinem Tagebuch.")

    diary = st.session_state.diary
    if not diary:
        st.info("Für deine Statistik brauchst du zuerst Tagebucheinträge.")
        empty_state("📈", "Deine Statistik entsteht hier",
                    "Je mehr Einträge du sammelst, desto interessanter wird dein Verlauf.")
        return

    df = pd.DataFrame(diary).drop(columns=["id", "Notiz"])
    df["Datum"] = pd.to_datetime(df["Datum"])
    df = df.sort_values("Datum")

    col1, col2, col3 = st.columns(3)
    with col1:
        stat_card(f"{df['Hautbewertung'].mean():.1f}/5", "Durchschnittliche Hautbewertung")
    with col2:
        stat_card(len(df), "Tagebucheinträge")
    with col3:
        stat_card(f"{df['Hautbewertung'].iloc[-1]}/5", "Letzte Bewertung")

    space()
    st.markdown("### 📈 Entwicklung deiner Hautbewertung")
    st.line_chart(df.set_index("Datum")[["Hautbewertung"]])
    st.markdown("### 🔎 Weitere Werte")
    st.line_chart(df.set_index("Datum")[["Trockenheit", "Rötungen", "Unreinheiten"]])
    st.caption("Die Statistik basiert nur auf deinen eigenen Tagebucheinträgen dieser Sitzung.")

# ============================================================
# SEITE: WIRKSTOFF-LEXIKON
# ============================================================


def page_actives():
    hero("🧪 Wirkstoff-Lexikon",
         "Entdecke wichtige Wirkstoffe der Hautpflege und erfahre, wofür sie eingesetzt werden.")

    search = st.text_input("Wirkstoff suchen", placeholder="z. B. Niacinamid oder Feuchtigkeit").strip().lower()
    selected = [(name, data) for name, data in ACTIVES.items()
                if not search or search in name.lower() or search in data[1].lower()]
    if not selected:
        st.info("Keinen passenden Wirkstoff gefunden.")
        return

    for name, (icon, category, description, benefits, tip) in selected:
        with st.expander(f"{icon} {name} · {category}"):
            html(f'<span class="badge">{category}</span><p style="margin-top:0.8rem;">{description}</p>')
            st.markdown("**Typische Eigenschaften**")
            for benefit in benefits:
                st.markdown(f"✓ {benefit}")
            st.info(f"💡 {tip}")

# ============================================================
# SEITE: QUIZ
# ============================================================


def quiz_check():
    question = QUIZ[st.session_state.quiz_index]
    answer = st.session_state.get(f"quiz_{st.session_state.quiz_index}")
    st.session_state.quiz_checked = True
    st.session_state.quiz_correct = answer == question["answer"]
    if st.session_state.quiz_correct:
        st.session_state.quiz_score += 1


def quiz_next():
    st.session_state.quiz_index += 1
    st.session_state.quiz_checked = False


def quiz_restart():
    for key in ("quiz_index", "quiz_score"):
        st.session_state[key] = 0
    st.session_state.quiz_checked = False


def page_quiz():
    hero("🎯 Hautpflege-Quiz", "Teste dein Wissen über Hautpflege und Wirkstoffe.")
    total = len(QUIZ)
    index = st.session_state.quiz_index

    if index >= total:
        card(f"""
        <div style="text-align:center;">
        <div style="font-size:3rem;">🏆</div>
        <div style="font-size:1.7rem; font-weight:700; color:#4A3328;">Quiz abgeschlossen!</div>
        <div style="font-size:1.1rem; color:#8D756A; margin-top:0.5rem;">
        Dein Ergebnis: <b>{st.session_state.quiz_score} / {total}</b>
        </div>
        </div>
        """)
        st.button("Quiz neu starten", on_click=quiz_restart)
        return

    question = QUIZ[index]
    st.progress(index / total, text=f"Frage {index + 1} von {total}")
    card(f"""
    <span class="badge">Frage {index + 1}</span>
    <h3 style="margin:0.8rem 0 0 0;">{question["q"]}</h3>
    """)

    checked = st.session_state.quiz_checked
    answer = st.radio("Antwort auswählen", question["options"], index=None,
                      key=f"quiz_{index}", label_visibility="collapsed", disabled=checked)

    if not checked:
        st.button("Antwort prüfen", on_click=quiz_check, disabled=answer is None)
        return

    if st.session_state.quiz_correct:
        st.success(f"Richtig! {question['why']}")
    else:
        st.error(f"Nicht ganz. Richtig ist: {question['answer']}. {question['why']}")
    st.button("Weiter" if index + 1 < total else "Ergebnis ansehen", on_click=quiz_next)

# ============================================================
# SEITE: INHALTSSTOFFE-CHECK
# ============================================================


def page_ingredients():
    hero("🔎 Inhaltsstoffe-Check",
         "Füge eine INCI-Liste ein und finde bekannte Wirkstoffe und ausgewählte Hinweise.")

    text = st.text_area("Zutatenliste", height=170,
                        placeholder="Aqua, Glycerin, Niacinamide, Parfum, Panthenol ...")
    if not text.strip():
        empty_state("🧪", "Zutatenliste hier einfügen",
                    "Die App sucht anschließend nach bekannten Begriffen.")
        return

    lowered = text.lower()
    found = [(keyword, icon, note) for keyword, (icon, note) in INGREDIENTS.items()
             if keyword in lowered]

    st.markdown("### Ergebnis")
    if not found:
        st.info("Keine Stoffe aus unserer kleinen Stichwortliste gefunden. "
                "Das heißt nicht, dass das Produkt für dich passt.")
        return

    for keyword, icon, note in found:
        html(f"""
        <div class="step">
        <div style="font-weight:650;">{icon} {keyword.title()}</div>
        <div class="step-desc">{note}</div>
        </div>
        """)
    st.caption("⚠️ möglicher Reizstoff · ℹ️ Wirkstoff mit Hinweisen · ✅ pflegend. "
               "Nur eine einfache Stichwortsuche, keine vollständige Produktbewertung.")

# ============================================================
# SEITE: INFO
# ============================================================


def page_about():
    hero("ℹ️ Über die SkinCareApp", "Informationen über Projekt, KI-Modell und Datenschutz.")
    card("""
    <h3>🌸 Unser Projekt</h3>
    <p>Die SkinCareApp ist ein Schulprojekt rund um Hautpflege, künstliche Intelligenz
    und digitale Gesundheitsbildung.</p>
    <p>Die App verbindet eine KI-Hautanalyse mit praktischen Funktionen wie Pflegeplan,
    Tagebuch, Statistik, Wirkstoff-Lexikon und Quiz.</p>
    """)
    card(f"""
    <h3>🤖 KI-Modell</h3>
    <p>Verwendet wird das vortrainierte Hugging-Face-Modell:</p>
    <span class="badge">{MODEL_ID}</span>
    <p class="small" style="margin-top:0.8rem;">
    Das Modell ordnet ein Bild genau einer von sieben Kategorien zu (Unreinheiten, Rötungen,
    Pigmentierung, Hautstruktur, Augenringe, geschwollene Augen, Falten). Die Genauigkeit wird
    von den Autoren mit etwa 57 % angegeben. Poren und trockene Haut erkennt es nicht.
    Die Ergebnisse können falsch sein.
    </p>
    """)
    card("""
    <h3>🔒 Datenschutz</h3>
    <p>Fotos werden nur im Arbeitsspeicher verarbeitet, nicht gespeichert und nicht an externe
    KI-Dienste gesendet. Das Modell läuft in der App selbst. Tagebuch, Verlauf und Quiz-Stand
    gelten nur für deine Sitzung und sind beim Schließen der Seite weg.</p>
    """)
    st.warning(DISCLAIMER)

# ============================================================
# NAVIGATION UND START
# ============================================================


PAGES = {
    "🏠 Startseite": page_home,
    "🤖 KI-Hautanalyse": page_analysis,
    "🧴 Pflegeplan": page_routine,
    "📔 Pflege-Tagebuch": page_diary,
    "📊 Haut-Statistik": page_statistics,
    "🧪 Wirkstoff-Lexikon": page_actives,
    "🎯 Hautpflege-Quiz": page_quiz,
    "🔎 Inhaltsstoffe-Check": page_ingredients,
    "ℹ️ Info & Datenschutz": page_about,
}

html("""
<div style="padding:0.7rem 0 1.2rem 0; text-align:center;">
<div style="font-size:2rem;">🌸</div>
<div style="font-size:1.25rem; font-weight:700; color:#4A3328;">SkinCareApp</div>
<div style="color:#967D72; font-size:0.75rem;">Deine persönliche Skincare-Welt</div>
</div>
""", sidebar=True)

selected_page = st.sidebar.radio("Navigation", list(PAGES))
PAGES[selected_page]()

html(f"""
<div style="margin-top:3rem; padding:1.2rem; text-align:center; color:#9B8278; font-size:0.75rem;">
🌸 SkinCareApp · Schulprojekt · {DISCLAIMER}
</div>
""")
