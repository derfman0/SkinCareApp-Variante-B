"""SkinCareApp - minimalistische Hautanalyse (Schulprojekt)."""

import io
from datetime import datetime

import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


# ============================================================
# EINSTELLUNGEN
# ============================================================

MODEL_ID = "varun1505/face-characteristics"
MAX_FILE_MB = 5
CONFIDENCE_THRESHOLD = 0.50

DISCLAIMER = (
    "Die SkinCareApp ist eine Lern-App zur Hautpflege. "
    "Sie stellt keine medizinische Diagnose und ersetzt keine ärztliche Beratung."
)


# ============================================================
# MODELL-KLASSEN
# ============================================================

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


# ============================================================
# PFLEGETIPPS
# ============================================================

CARE = {
    "Hautunreinheiten": (
        "Sichtbare Unreinheiten wie Pickel oder Mitesser.",
        [
            "Zweimal täglich sanft reinigen, nicht schrubben.",
            "Nicht ausdrücken, das kann Reizungen und Narben fördern.",
            "Produkte wählen, die als 'nicht komedogen' gekennzeichnet sind.",
            "Bei starken oder schmerzhaften Unreinheiten: Hautarzt fragen.",
        ],
    ),
    "Rötungen": (
        "Sichtbare Rötungen der Haut.",
        [
            "Milde, parfümfreie Produkte verwenden.",
            "Sehr heißes Wasser und grobe Peelings meiden.",
            "Die Pflegeroutine einfach halten.",
            "Bei anhaltender Rötung: Hautarzt fragen.",
        ],
    ),
    "Pigmentierung": (
        "Sichtbare dunklere Flecken auf der Haut.",
        [
            "Täglich Sonnenschutz (mindestens LSF 30) verwenden.",
            "Geduld haben: Hautveränderungen brauchen Zeit.",
            "Neue oder sich verändernde Flecken immer ärztlich anschauen lassen.",
        ],
    ),
    "Hautstruktur": (
        "Eine auffällige Oberflächenstruktur der Haut.",
        [
            "Sanft reinigen und regelmäßig eincremen.",
            "Nicht zu oft peelen, das reizt die Hautbarriere.",
            "Täglich Sonnenschutz verwenden.",
        ],
    ),
    "Augenringe": (
        "Dunkle Schatten unter den Augen. Sie sind oft erblich bedingt.",
        [
            "Ausreichend schlafen.",
            "Sonnenschutz und eine milde Augenpflege verwenden.",
            "Ein kühles Tuch kann sich erfrischend anfühlen.",
        ],
    ),
    "Geschwollene Augen": (
        "Sichtbare Schwellungen im Augenbereich.",
        [
            "Genug trinken und ausreichend schlafen.",
            "Ein kühles Tuch kann helfen.",
            "Bei dauerhaften Schwellungen: Arzt fragen.",
        ],
    ),
    "Falten": (
        "Sichtbare Linien und Falten. Sie gehören zum natürlichen Altern.",
        [
            "Täglich Sonnenschutz ist die wichtigste Basis.",
            "Regelmäßig Feuchtigkeitspflege verwenden.",
            "Kein Produkt kann Falten garantiert entfernen.",
        ],
    ),
}


# ============================================================
# INHALTSSTOFFE
# ============================================================

INGREDIENTS = {
    "parfum": (
        "⚠️",
        "Duftstoff: kann empfindliche Haut reizen.",
    ),
    "fragrance": (
        "⚠️",
        "Duftstoff: kann empfindliche Haut reizen.",
    ),
    "limonene": (
        "⚠️",
        "Duftstoff-Allergen: kann bei empfindlicher Haut reizen.",
    ),
    "linalool": (
        "⚠️",
        "Duftstoff-Allergen: kann bei empfindlicher Haut reizen.",
    ),
    "alcohol denat": (
        "⚠️",
        "Vergällter Alkohol: kann austrocknen und reizen.",
    ),
    "sodium lauryl sulfate": (
        "⚠️",
        "Starkes Reinigungsmittel: kann die Haut austrocknen.",
    ),
    "menthol": (
        "⚠️",
        "Kann bei empfindlicher Haut reizen.",
    ),
    "retinol": (
        "ℹ️",
        "Starker Wirkstoff: langsam einführen, Sonnenschutz nutzen, "
        "in der Schwangerschaft ärztlich klären.",
    ),
    "salicylic acid": (
        "ℹ️",
        "Säure (BHA): kann bei Unreinheiten genutzt werden, "
        "aber auch reizen. Vorsichtig dosieren.",
    ),
    "glycolic acid": (
        "ℹ️",
        "Säure (AHA): macht die Haut sonnenempfindlicher. Sonnenschutz nutzen.",
    ),
    "niacinamide": (
        "✅",
        "Wird meist gut vertragen.",
    ),
    "glycerin": (
        "✅",
        "Feuchtigkeitsspendend.",
    ),
    "hyaluronic": (
        "✅",
        "Feuchtigkeitsspendend.",
    ),
    "ceramide": (
        "✅",
        "Unterstützt die Hautbarriere.",
    ),
    "panthenol": (
        "✅",
        "Beruhigend und feuchtigkeitsspendend.",
    ),
}


# ============================================================
# STREAMLIT-SEITENEINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="SkinCareApp",
    page_icon="🌸",
    layout="centered",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>
    #MainMenu, footer {
        visibility: hidden;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 720px;
    }

    h1, h2, h3 {
        color: #4A3328;
        font-weight: 600;
        letter-spacing: -0.01em;
    }

    .card {
        background: #FFFFFF;
        border-radius: 18px;
        padding: 1.1rem 1.3rem;
        box-shadow: 0 2px 12px rgba(74, 51, 40, 0.07);
        margin-bottom: 1rem;
    }

    .small {
        color: #8a7468;
        font-size: 0.85rem;
    }

    .stButton > button {
        border-radius: 999px;
        border: none;
        background: #D8A7A0;
        color: white;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    .stButton > button:hover {
        background: #c8938b;
        color: white;
    }

    .stProgress > div > div > div > div {
        background-color: #D8A7A0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource(
    show_spinner="KI-Modell wird geladen (der erste Start kann etwas dauern) ..."
)
def load_model():
    """
    Lädt Bildprozessor und Modell einmalig.

    WICHTIG:
    Wir verwenden hier ViTImageProcessor anstelle von
    AutoImageProcessor, weil das Modell ein ViT-Modell ist.
    """

    from transformers import (
        ViTImageProcessor,
        AutoModelForImageClassification,
    )

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)

    model.eval()

    return processor, model


# ============================================================
# BILD KLASSIFIZIEREN
# ============================================================

def classify(image):
    """
    Analysiert ein Bild und gibt eine nach Wahrscheinlichkeit
    sortierte Liste zurück.

    Format:
    (deutscher Name, Original-Label, Score)
    """

    import torch

    processor, model = load_model()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(logits, dim=-1)[0]

    results = []

    for index, score in enumerate(probabilities.tolist()):

        original = model.config.id2label[index]

        key = (
            original
            .replace("_", " ")
            .replace("-", " ")
            .strip()
            .lower()
        )

        german_name = LABEL_MAP.get(key, original)

        results.append(
            (
                german_name,
                original,
                float(score),
            )
        )

    return sorted(
        results,
        key=lambda result: result[2],
        reverse=True,
    )


# ============================================================
# BILD-HILFSFUNKTIONEN
# ============================================================

def load_valid_image(file):
    """
    Prüft Dateigröße und Bildformat.
    Gibt (Bild, Fehlertext) zurück.
    """

    data = file.getvalue()

    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return (
            None,
            f"Die Datei ist größer als {MAX_FILE_MB} MB. "
            "Bitte wähle ein kleineres Foto.",
        )

    try:
        Image.open(io.BytesIO(data)).verify()

        image = Image.open(io.BytesIO(data))

        image = (
            ImageOps.exif_transpose(image)
            .convert("RGB")
        )

        return image, None

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return (
            None,
            "Diese Datei konnte nicht als Bild gelesen werden. "
            "Bitte versuche ein anderes Foto.",
        )


# ============================================================
# FOTO-QUALITÄT
# ============================================================

def check_photo_quality(image):
    """
    Grobe Prüfung von Helligkeit und Schärfe.
    """

    gray = np.asarray(
        image.convert("L").resize((256, 256)),
        dtype=float,
    )

    brightness = gray.mean()

    laplace = (
        -4 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1]
        + gray[2:, 1:-1]
        + gray[1:-1, :-2]
        + gray[1:-1, 2:]
    )

    sharpness = laplace.var()

    hints = []

    if brightness < 70:
        hints.append(
            "Das Foto ist ziemlich dunkel. "
            "Nimm es bei Tageslicht auf."
        )

    if brightness > 200:
        hints.append(
            "Das Foto ist sehr hell. "
            "Vermeide direktes, grelles Licht."
        )

    if sharpness < 30:
        hints.append(
            "Das Foto wirkt unscharf. "
            "Halte die Kamera ruhig."
        )

    return hints


# ============================================================
# BILD ZURÜCKSETZEN
# ============================================================

def reset_image():
    """
    Entfernt das aktuelle Bild.
    """

    st.session_state.reset_count = (
        st.session_state.get("reset_count", 0) + 1
    )

    st.session_state.result = None


# ============================================================
# BERICHT
# ============================================================

def build_report(results):
    """
    Erstellt einen Textbericht ohne Foto.
    """

    top_name, _, top_score = results[0]

    lines = [
        "SkinCareApp - Analysebericht",
        datetime.now().strftime(
            "Datum: %d.%m.%Y %H:%M"
        ),
        "",
        f"Wahrscheinlichstes Merkmal: "
        f"{top_name} ({top_score:.0%})",
        "",
        "Alle Modellwerte:",
    ]

    lines += [
        f"- {name}: {score:.0%}"
        for name, _, score in results
    ]

    lines += [
        "",
        f"Modell: {MODEL_ID}",
        DISCLAIMER,
    ]

    return "\n".join(lines)


# ============================================================
# SEITE: HAUTANALYSE
# ============================================================

def page_analysis():

    st.subheader("Hautanalyse")

    st.markdown(
        """
        <p class='small'>
        Dein Foto wird nur in dieser App verarbeitet,
        nicht gespeichert und nicht an einen externen
        KI-Dienst gesendet.
        </p>
        """,
        unsafe_allow_html=True,
    )

    n = st.session_state.get("reset_count", 0)

    source = st.radio(
        "Bildquelle",
        [
            "Foto hochladen",
            "Selfie aufnehmen",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    # --------------------------------------------------------
    # FOTO HOCHLADEN
    # --------------------------------------------------------

    if source == "Foto hochladen":

        file = st.file_uploader(
            "Selfie hochladen (JPG, JPEG, PNG)",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key=f"upload_{n}",
        )

    # --------------------------------------------------------
    # KAMERA
    # --------------------------------------------------------

    else:

        st.caption(
            "Dein Browser fragt nach der Erlaubnis "
            "für die Kamera. Falls die Kamera nicht "
            "funktioniert, nutze den Foto-Upload."
        )

        file = st.camera_input(
            "Selfie aufnehmen",
            key=f"camera_{n}",
        )

    if file is None:
        return

    # --------------------------------------------------------
    # BILD PRÜFEN
    # --------------------------------------------------------

    image, error = load_valid_image(file)

    if error:

        st.error(error)

        return

    st.image(
        image,
        caption="Ausgewähltes Foto",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # FOTOQUALITÄT
    # --------------------------------------------------------

    for hint in check_photo_quality(image):
        st.warning(hint)

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    col1.button(
        "Bild entfernen",
        on_click=reset_image,
    )

    analyze = col2.button(
        "Haut analysieren",
    )

    # --------------------------------------------------------
    # KI-ANALYSE
    # --------------------------------------------------------

    if analyze:

        try:

            with st.spinner(
                "Deine Haut wird analysiert ..."
            ):

                results = classify(image)

            st.session_state.result = results

            history = st.session_state.get(
                "history",
                [],
            )

            history.append(
                {
                    "Zeit": datetime.now().strftime(
                        "%H:%M:%S"
                    ),
                    "Ergebnis": results[0][0],
                    "Sicherheit": f"{results[0][2]:.0%}",
                }
            )

            st.session_state.history = history

        except Exception as error_info:

            st.session_state.result = None

            st.error(
                "Das KI-Modell konnte nicht geladen "
                "oder ausgeführt werden. Bitte versuche "
                "es später noch einmal."
            )

            with st.expander(
                "Technische Details"
            ):

                st.code(
                    f"{type(error_info).__name__}: "
                    f"{error_info}"
                )

    # --------------------------------------------------------
    # ERGEBNIS ANZEIGEN
    # --------------------------------------------------------

    if st.session_state.get("result"):

        show_results(
            st.session_state.result
        )


# ============================================================
# ERGEBNISSE
# ============================================================

def show_results(results):

    st.subheader("Ergebnis")

    top_name, _, top_score = results[0]

    # --------------------------------------------------------
    # NICHT SICHER
    # --------------------------------------------------------

    if top_score < CONFIDENCE_THRESHOLD:

        st.warning(
            "Wir konnten dieses Bild nicht sicher "
            "auswerten. Bitte versuche ein anderes Foto."
        )

        st.markdown(
            f"""
            <p class='small'>
            Kategorie: Unklar / keine Erkennung
            (höchster Modellwert {top_score:.0%},
            Schwelle {CONFIDENCE_THRESHOLD:.0%}).
            </p>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # ERGEBNIS
    # --------------------------------------------------------

    else:

        explanation, tips = CARE.get(
            top_name,
            ("", []),
        )

        st.markdown(
            f"""
            <div class='card'>
            <b>Wahrscheinlichstes sichtbares Merkmal:</b>
            {top_name}
            <br>

            <span class='small'>
            {explanation}

            Sicherheit des Modells:
            {top_score:.0%}.

            Das Modell wählt
            <u>eine</u> Kategorie pro Bild und
            erkennt nicht mehrere Merkmale gleichzeitig.
            </span>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if tips:

            st.markdown(
                "**Allgemeine Pflegetipps**"
            )

            for tip in tips:
                st.markdown(
                    f"- {tip}"
                )

    # --------------------------------------------------------
    # ALLE MODELLWERTE
    # --------------------------------------------------------

    st.markdown(
        "**Alle Modellwerte**"
    )

    for name, _, score in results:

        st.progress(
            min(max(score, 0.0), 1.0),
            text=f"{name}: {score:.0%}",
        )

    st.markdown(
        f"""
        <p class='small'>
        Von diesem Modell nicht erkannt:
        {', '.join(NOT_SUPPORTED)}.

        Die Werte sind die Ausgabe des Modells
        und kein medizinisches Maß.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # BERICHT DOWNLOAD
    # --------------------------------------------------------

    st.download_button(
        "Bericht herunterladen (ohne Foto)",
        build_report(results),
        file_name="skincare_bericht.txt",
    )


# ============================================================
# SEITE: VERLAUF
# ============================================================

def page_history():

    st.subheader("Verlauf")

    st.caption(
        "Nur für diese Sitzung. "
        "Es werden keine Fotos gespeichert."
    )

    history = st.session_state.get(
        "history",
        [],
    )

    if not history:

        st.info(
            "Noch keine Analyse in dieser Sitzung."
        )

        return

    st.table(history)

    if st.button(
        "Verlauf löschen"
    ):

        st.session_state.history = []

        st.rerun()


# ============================================================
# SEITE: HAUTTYP-QUIZ
# ============================================================

def page_quiz():

    st.subheader("Hauttyp-Quiz")

    st.caption(
        "Zwei Fragen für allgemeine Hinweise. "
        "Keine Diagnose."
    )

    sun = st.radio(
        "Wie reagiert deine Haut auf Sonne ohne Schutz?",
        [
            "Verbrennt schnell, bräunt kaum",
            "Verbrennt oft, bräunt langsam",
            "Verbrennt manchmal, bräunt gut",
            "Verbrennt selten, bräunt schnell",
        ],
        index=None,
    )

    feel = st.radio(
        "Wie fühlt sich deine Haut mittags an?",
        [
            "Spannt oder fühlt sich trocken an",
            "Normal und angenehm",
            "Glänzt, vor allem an Stirn und Nase",
        ],
        index=None,
    )

    if sun is None or feel is None:
        return

    st.markdown(
        "**Deine Hinweise**"
    )

    if sun in (
        "Verbrennt schnell, bräunt kaum",
        "Verbrennt oft, bräunt langsam",
    ):

        st.write(
            "☀️ Deine Haut ist sonnenempfindlich: "
            "Nutze täglich Sonnenschutz mit LSF 50, "
            "meide die Mittagssonne und creme bei Bedarf nach."
        )

    else:

        st.write(
            "☀️ Auch bei weniger sonnenempfindlicher Haut gilt: "
            "Täglich Sonnenschutz mit mindestens LSF 30 "
            "schützt vor Hautschäden."
        )

    if feel.startswith("Spannt"):

        st.write(
            "💧 Trockene Tendenz: Milde Reinigung, "
            "lauwarmes Wasser und eine reichhaltigere "
            "Feuchtigkeitscreme können helfen."
        )

    elif feel.startswith("Normal"):

        st.write(
            "💧 Normale Haut: Sanft reinigen, "
            "eincremen und Sonnenschutz genügen meist."
        )

    else:

        st.write(
            "💧 Ölige Tendenz: Leichte, nicht komedogene "
            "Pflege (z. B. Gel-Creme) passt oft gut. "
            "Trotzdem eincremen, auch fettige Haut "
            "braucht Feuchtigkeit."
        )

    st.markdown(
        f"""
        <p class='small'>
        {DISCLAIMER}
        </p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SEITE: INHALTSSTOFFE
# ============================================================

def page_ingredients():

    st.subheader("Inhaltsstoffe-Check")

    st.caption(
        "Füge die Zutatenliste (INCI) eines Pflegeprodukts ein."
    )

    text = st.text_area(
        "Zutatenliste",
        height=150,
        placeholder=(
            "Aqua, Glycerin, Parfum, "
            "Alcohol Denat., Niacinamide ..."
        ),
    )

    if not text.strip():
        return

    lowered = text.lower()

    found = []

    shown = set()

    for keyword, (icon, note) in INGREDIENTS.items():

        if (
            keyword in lowered
            and (icon, note) not in shown
        ):

            shown.add(
                (icon, note)
            )

            found.append(
                (keyword, icon, note)
            )

    if not found:

        st.info(
            "Keine Stoffe aus unserer kleinen Liste gefunden. "
            "Das heißt nicht, dass das Produkt für dich passend ist."
        )

    for keyword, icon, note in found:

        st.markdown(
            f"{icon} **{keyword.title()}**: {note}"
        )

    st.markdown(
        """
        <p class='small'>
        ⚠️ möglicher Reizstoff ·
        ℹ️ Wirkstoff mit Hinweisen ·
        ✅ pflegend.

        Das ist nur eine einfache Stichwortliste
        und keine Bewertung der Verträglichkeit.
        </p>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SEITE: INFO
# ============================================================

def page_about():

    st.subheader("Info & Datenschutz")

    st.write(
        "Die SkinCareApp ist ein Schulprojekt. "
        "Sie nutzt ein vortrainiertes KI-Modell von "
        f"Hugging Face (`{MODEL_ID}`), das ein Selfie "
        "einer sichtbaren Hautmerkmal-Kategorie zuordnet."
    )

    st.markdown(
        "**Grenzen des Modells**"
    )

    st.write(
        "Es wählt nur eine Kategorie pro Bild, "
        "gibt eine selbst angegebene Genauigkeit von "
        "etwa 57 % an und erkennt keine Poren und "
        "keine trockene Haut. Ergebnisse können falsch sein."
    )

    st.markdown(
        "**Datenschutz**"
    )

    st.write(
        "Fotos werden nur im Arbeitsspeicher verarbeitet, "
        "nicht gespeichert und nicht an externe KI-Dienste "
        "gesendet. Mit „Bild entfernen“ löschst du das "
        "aktuelle Foto aus der App."
    )

    st.info(DISCLAIMER)


# ============================================================
# START DER APP
# ============================================================

st.title("🌸 SkinCareApp")

st.caption(
    "Sanfte, KI-gestützte Einblicke "
    "in sichtbare Hautmerkmale."
)


tabs = st.tabs(
    [
        "Analyse",
        "Verlauf",
        "Hauttyp-Quiz",
        "Inhaltsstoffe",
        "Info",
    ]
)Die entscheidende Stelle in deiner app.py ist:
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

Das Modell varun1505/face-characteristics ist zwar auf Hugging Face als Image-Classification-Modell veröffentlicht und die Modellseite zeigt ebenfalls genau diesen Ladeweg. Aber deine aktuelle transformers-Version kann die vorhandene preprocessor_config.json offenbar nicht automatisch einem Bildprozessor zuordnen. Genau deshalb kommt:

    ValueError: Unrecognized image processor ... Should have a image_processor_type key...

Hugging Face erklärt, dass AutoImageProcessor normalerweise anhand der Processor-Konfiguration bzw. des Modelltyps entscheidet, welcher Bildprozessor verwendet werden soll.
Die einfachste Lösung

Da dein Modell ein ViT-Modell ist, würde ich den automatischen AutoImageProcessor umgehen und den für ViT passenden Processor explizit verwenden.

In deiner app.py musst du zunächst diesen Import ändern.

Aktuell:
from transformers import AutoImageProcessor, AutoModelForImageClassification

Ändern zu:
from transformers import ViTImageProcessor, AutoModelForImageClassification

Und diese Zeile:
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

ändern zu:
processor = ViTImageProcessor.from_pretrained(MODEL_ID)

Deine komplette load_model()-Funktion sollte dann so aussehen:
@st.cache_resource(show_spinner="KI-Modell wird geladen (der erste Start kann etwas dauern) ...")
def load_model():
    """Lädt Bildprozessor und Modell einmalig und merkt sie sich."""
    from transformers import ViTImageProcessor, AutoModelForImageClassification

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageClassification.from_pretrained(MODEL_ID)
    model.eval()

    return processor, model

Das ist die erste Änderung, die ich ausprobieren würde.
Aber: requirements.txt würde ich ebenfalls ändern

Deine aktuelle Datei ist:
--extra-index-url https://download.pytorch.org/whl/cpu
streamlit>=1.40
transformers>=4.44
torch
torchvision
pillow
numpy

Das >= bei transformers ist etwas problematisch für ein Schulprojekt, weil Streamlit beim nächsten Deployment eine deutlich neuere Transformers-Version installieren kann. Dein Modell wurde offenbar mit einer älteren/anderen Konfiguration veröffentlicht.

Ich würde deshalb zunächst eine konkrete Version verwenden:
--extra-index-url https://download.pytorch.org/whl/cpu

streamlit>=1.40,<2.0
transformers==4.44.2
torch
torchvision
pillow
numpy

Wichtig: Nicht einfach blind transformers auf die allerneueste Version setzen. Dein Fehler ist gerade ein Hinweis darauf, dass die automatische Erkennung des Processors zwischen Modell-Repository und installierter Transformers-Version nicht sauber zusammenspielt.

Die Hugging-Face-Dokumentation bestätigt außerdem, dass ein Image Processor die Bildgrößen, Normalisierung und Umwandlung in Tensoren übernimmt – genau das brauchst du vor deinem model(**inputs)-Aufruf.
Noch eine Sache in deinem Code

Du hast in deiner README.md geschrieben:

    varun1505/face-characteristics – Vision Transformer, ca. 86 Mio. Parameter.

Auf der aktuellen Hugging-Face-Seite ist das Repository weiterhin als ViT Image Classification ausgewiesen.

Das passt also zu ViTImageProcessor.
Was du jetzt konkret machen solltest

1. app.py ändern

Nur diesen Teil:
from transformers import AutoImageProcessor, AutoModelForImageClassification

zu:
from transformers import ViTImageProcessor, AutoModelForImageClassification

und:
processor = AutoImageProcessor.from_pretrained(MODEL_ID)

zu:
processor = ViTImageProcessor.from_pretrained(MODEL_ID)

2. requirements.txt ersetzen durch:
--extra-index-url https://download.pytorch.org/whl/cpu
streamlit>=1.40,<2.0
transformers==4.44.2
torch
torchvision
pillow
numpy

3. Commit + Push zu GitHub

Dann Streamlit Cloud neu deployen bzw. die App neu starten.
Falls danach ein neuer Fehler kommt

Dann schick mir den neuen Fehlertext. Besonders wichtig wäre dann die erste Zeile des Fehlers und die letzten ~10 Zeilen.

Denn es gibt bei diesem Modell noch einen zweiten Punkt, den wir gegebenenfalls prüfen müssen: die tatsächliche preprocessor_config.json des Modells. Das Repository enthält diese Datei tatsächlich.

Deine übrige App – Upload, Kamera, Foto-Qualitätscheck, Verlauf, Quiz und Inhaltsstoffe – muss dafür nicht neu geschrieben werden. Die problematische Stelle ist aktuell sehr gezielt der Modell-/Processor-Ladevorgang.



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


st.markdown(
    f"""
    <p class='small'>
    {DISCLAIMER}
    </p>
    """,
    unsafe_allow_html=True,
)
```
