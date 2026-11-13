"""
SkinCareApp
Moderne Skincare-App mit KI-Hautanalyse, Pflegeplan, Tagebuch,
Haut-Statistik, Wirkstoff-Lexikon und Hautpflege-Quiz.
"""

import io
import sqlite3
from datetime import datetime, date

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

DB_FILE = "skincare.db"


# ============================================================
# MODELL-KATEGORIEN
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

NOT_SUPPORTED = [
    "Poren",
    "Trockene Haut",
]


# ============================================================
# PFLEGETIPPS
# ============================================================

CARE = {
    "Hautunreinheiten": (
        "Sichtbare Unreinheiten wie Pickel oder Mitesser.",
        [
            "Zweimal täglich sanft reinigen.",
            "Nicht ausdrücken, da dies die Haut reizen kann.",
            "Nicht komedogene Pflege bevorzugen.",
            "Bei starken oder schmerzhaften Unreinheiten Hautarzt fragen.",
        ],
    ),

    "Rötungen": (
        "Sichtbare Rötungen der Haut.",
        [
            "Milde und möglichst parfümfreie Produkte verwenden.",
            "Sehr heißes Wasser und grobe Peelings vermeiden.",
            "Die Pflegeroutine möglichst einfach halten.",
            "Bei anhaltender Rötung ärztlich abklären lassen.",
        ],
    ),

    "Pigmentierung": (
        "Sichtbare dunklere Flecken oder ungleichmäßige Pigmentierung.",
        [
            "Täglich Sonnenschutz verwenden.",
            "Geduldig sein, da Hautveränderungen Zeit benötigen.",
            "Neue oder sich verändernde Flecken ärztlich anschauen lassen.",
        ],
    ),

    "Hautstruktur": (
        "Eine auffällige oder ungleichmäßige Oberflächenstruktur.",
        [
            "Sanft reinigen und regelmäßig Feuchtigkeitspflege verwenden.",
            "Nicht zu häufig peelen.",
            "Täglich Sonnenschutz verwenden.",
        ],
    ),

    "Augenringe": (
        "Dunkle Schatten unter den Augen.",
        [
            "Ausreichend schlafen.",
            "Eine milde Augenpflege verwenden.",
            "Sonnenschutz auch im Bereich um die Augen beachten.",
        ],
    ),

    "Geschwollene Augen": (
        "Sichtbare Schwellungen im Augenbereich.",
        [
            "Ausreichend schlafen und trinken.",
            "Ein kühles Tuch kann angenehm sein.",
            "Bei dauerhaften Schwellungen ärztlich abklären lassen.",
        ],
    ),

    "Falten": (
        "Sichtbare Linien und Falten, die zum natürlichen Altern gehören.",
        [
            "Täglich Sonnenschutz verwenden.",
            "Regelmäßig Feuchtigkeitspflege verwenden.",
            "Keine Pflege kann Falten garantiert entfernen.",
        ],
    ),
}


# ============================================================
# WIRKSTOFF-LEXIKON
# ============================================================

ACTIVES = {
    "Niacinamid": {
        "icon": "💧",
        "category": "Beruhigend",
        "description": (
            "Niacinamid ist eine Form von Vitamin B3 und wird häufig "
            "in Pflegeprodukten eingesetzt."
        ),
        "benefits": [
            "Unterstützt die Hautbarriere",
            "Kann Rötungen beruhigen",
            "Kann das Hautbild ausgleichen",
        ],
        "tip": "Besonders interessant bei unreiner oder empfindlicher Haut.",
    },

    "Hyaluronsäure": {
        "icon": "💦",
        "category": "Feuchtigkeit",
        "description": (
            "Hyaluronsäure ist ein feuchtigkeitsbindender Wirkstoff "
            "und Bestandteil vieler Feuchtigkeitsprodukte."
        ),
        "benefits": [
            "Unterstützt die Feuchtigkeitsversorgung",
            "Kann die Haut geschmeidiger wirken lassen",
            "Passt zu vielen Hauttypen",
        ],
        "tip": "Gut mit einer anschließenden Creme kombinierbar.",
    },

    "Retinol": {
        "icon": "✨",
        "category": "Anti-Aging",
        "description": (
            "Retinol gehört zur Gruppe der Vitamin-A-Derivate "
            "und wird in der Hautpflege eingesetzt."
        ),
        "benefits": [
            "Unterstützt die Hauterneuerung",
            "Kann das Hautbild verbessern",
            "Wird häufig in Anti-Aging-Pflege eingesetzt",
        ],
        "tip": "Langsam einführen und tagsüber konsequent Sonnenschutz verwenden.",
    },

    "Salicylsäure": {
        "icon": "🫧",
        "category": "BHA",
        "description": (
            "Salicylsäure ist eine Beta-Hydroxysäure und wird häufig "
            "bei unreiner Haut eingesetzt."
        ),
        "benefits": [
            "Kann Porenbereiche von überschüssigem Talg befreien",
            "Wird häufig bei Unreinheiten eingesetzt",
            "Kann die Hautoberfläche glätten",
        ],
        "tip": "Bei empfindlicher Haut langsam und vorsichtig einsetzen.",
    },

    "Panthenol": {
        "icon": "🌿",
        "category": "Beruhigend",
        "description": (
            "Panthenol wird häufig in beruhigenden und pflegenden "
            "Produkten verwendet."
        ),
        "benefits": [
            "Unterstützt die Hautpflege",
            "Kann beruhigend wirken",
            "Hilft bei trockener oder beanspruchter Haut",
        ],
        "tip": "Besonders beliebt in beruhigenden Pflegeprodukten.",
    },

    "Ceramide": {
        "icon": "🛡️",
        "category": "Hautbarriere",
        "description": (
            "Ceramide sind natürliche Bestandteile der Hautbarriere "
            "und kommen auch in Pflegeprodukten vor." Startseite
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
        ), Startseite Startseite Startseite Startseite
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
        "benefits": [ Startseite
🤖 KI-Hautanalyse
🧴 Persönlicher Pflegeplan
📔 Pflege-Tagebuch
📊 Haut-Statistik
🧪 Wirkstoff-Lexikon
🎯 Hautpflege-Quiz     Ich soll als erstes Layout von meinem App vorstellen
            "Unterstützen die Hautbarriere",
            "Helfen, Feuchtigkeit in der Haut zu halten",
            "Interessant bei trockener Haut",
        ],
        "tip": "Eine gute Ergänzung für eine einfache Barriereroutine.",
    },

    "Glycerin": {
        "icon": "💧",
        "category": "Feuchtigkeit",
        "description": (
            "Glycerin ist ein Feuchthaltemittel und kommt in vielen "
            "Cremes und Lotionen vor."
        ),
        "benefits": [
            "Bindet Feuchtigkeit",
            "Unterstützt ein geschmeidiges Hautgefühl",
            "Wird von vielen Hauttypen gut vertragen",
        ],
        "tip": "Ein sehr häufiger und vielseitiger Pflegebestandteil.",
    },
}


# ============================================================
# INGREDIENT CHECKER
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
        "Vergällter Alkohol: kann bei empfindlicher Haut austrocknend oder reizend wirken.",
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
        "Starker Wirkstoff: langsam einführen und tagsüber Sonnenschutz verwenden.",
    ),
    "salicylic acid": (
        "ℹ️",
        "BHA-Säure: wird häufig bei Unreinheiten eingesetzt, kann aber auch reizen.",
    ),
    "glycolic acid": (
        "ℹ️",
        "AHA-Säure: kann die Haut empfindlicher gegenüber Sonne machen.",
    ),
    "niacinamide": (
        "✅",
        "Wird meist gut vertragen und wird häufig in Pflegeprodukten verwendet.",
    ),
    "glycerin": (
        "✅",
        "Feuchtigkeitsspendender Inhaltsstoff.",
    ),
    "hyaluronic": (
        "✅",
        "Feuchtigkeitsspendender Inhaltsstoff.",
    ),
    "ceramide": (
        "✅",
        "Unterstützt die Hautbarriere.",
    ),
    "panthenol": (
        "✅",
        "Wird häufig für beruhigende und pflegende Produkte verwendet.",
    ),
}


# ============================================================
# STREAMLIT EINSTELLUNGEN
# ============================================================

st.set_page_config(
    page_title="SkinCareApp",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN
# ============================================================

st.markdown(
    """
    <style>

    /* ------------------------------
       GLOBAL
    ------------------------------ */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(232, 193, 187, 0.25),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(216, 167, 160, 0.18),
                transparent 28%
            ),
            radial-gradient(
                circle at 50% 90%,
                rgba(244, 229, 214, 0.35),
                transparent 35%
            ),
            #FBF8F3;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ------------------------------
       SIDEBAR
    ------------------------------ */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #F4E9E1 0%,
                #FBF8F3 55%,
                #F7EEE8 100%
            );
        border-right: 1px solid rgba(130, 90, 80, 0.08);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.4rem;
    }

    /* ------------------------------
       TEXT
    ------------------------------ */

    h1, h2, h3 {
        color: #4A3328 !important;
        font-weight: 600 !important;
        letter-spacing: -0.025em;
    }

    p, label, span {
        color: #5D463C;
    }

    .small {
        color: #947C70;
        font-size: 0.86rem;
    }

    /* ------------------------------
       BUTTONS
    ------------------------------ */

    .stButton > button {
        border-radius: 999px;
        border: 1px solid #D8A7A0;
        background: #D8A7A0;
        color: white;
        font-weight: 600;
        padding: 0.55rem 1.35rem;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: #C99189;
        border-color: #C99189;
        color: white;
        transform: translateY(-1px);
    }

    /* ------------------------------
       CARDS
    ------------------------------ */

    .hero {
        background:
            linear-gradient(
                135deg,
                rgba(255,255,255,0.96),
                rgba(249,235,230,0.92)
            );
        border: 1px solid rgba(150, 105, 95, 0.12);
        border-radius: 28px;
        padding: 2.2rem 2.5rem;
        box-shadow: 0 15px 45px rgba(93, 60, 50, 0.08);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 35px;
        top: 20px;
        font-size: 4rem;
        color: rgba(216, 167, 160, 0.18);
    }

    .hero-title {
        font-size: 2.8rem;
        font-weight: 650;
        color: #4A3328;
        margin-bottom: 0.4rem;
    }

    .hero-subtitle {
        font-size: 1.05rem;
        color: #846D62;
        max-width: 680px;
        line-height: 1.7;
    }

    .section-card {
        background: rgba(255,255,255,0.78);
        border: 1px solid rgba(140, 100, 90, 0.10);
        border-radius: 22px;
        padding: 1.3rem;
        box-shadow: 0 8px 28px rgba(93, 60, 50, 0.05);
        height: 100%;
    }

    .feature-card {
        background: rgba(255,255,255,0.88);
        border: 1px solid rgba(140, 100, 90, 0.10);
        border-radius: 20px;
        padding: 1.2rem;
        min-height: 170px;
        box-shadow: 0 8px 24px rgba(93, 60, 50, 0.05);
    }

    .feature-icon {
        width: 42px;
        height: 42px;
        border-radius: 14px;
        background: #F5E1DB;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        margin-bottom: 0.7rem;
    }

    .feature-title {
        color: #4A3328;
        font-weight: 650;
        font-size: 1.05rem;
    }

    .feature-text {
        color: #89736A;
        font-size: 0.88rem;
        line-height: 1.5;
        margin-top: 0.35rem;
    }

    /* ------------------------------
       BADGES
    ------------------------------ */

    .badge {
        display: inline-block;
        background: #F4E0DA;
        color: #7E554C;
        border-radius: 999px;
        padding: 0.3rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
    }

    .badge-green {
        background: #E6EFE5;
        color: #55704F;
    }

    .badge-blue {
        background: #E7EEF2;
        color: #566D78;
    }

    .badge-purple {
        background: #EEE7F2;
        color: #6E5D78;
    }

    /* ------------------------------
       STATISTICS
    ------------------------------ */

    .stat-card {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(140, 100, 90, 0.10);
        border-radius: 20px;
        padding: 1.15rem;
        text-align: center;
        box-shadow: 0 7px 22px rgba(93, 60, 50, 0.05);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #4A3328;
    }

    .stat-label {
        color: #967D72;
        font-size: 0.8rem;
        margin-top: 0.15rem;
    }

    /* ------------------------------
       CHECKLIST
    ------------------------------ */

    .routine-step {
        background: #FFFDFC;
        border: 1px solid #EEDDD7;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.7rem;
    }

    .routine-title {
        font-weight: 650;
        color: #4A3328;
    }

    .routine-desc {
        color: #927A70;
        font-size: 0.82rem;
    }

    /* ------------------------------
       QUOTE
    ------------------------------ */

    .quote {
        background: linear-gradient(
            135deg,
            #F5E5DF,
            #FAF4EF
        );
        border-radius: 20px;
        padding: 1.2rem 1.4rem;
        color: #76594E;
        font-style: italic;
        border-left: 4px solid #D8A7A0;
    }

    /* ------------------------------
       PROGRESS
    ------------------------------ */

    .stProgress > div > div > div > div {
        background-color: #D8A7A0;
    }

    /* ------------------------------
       INPUTS
    ------------------------------ */

    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        border-radius: 14px !important;
        border-color: #E6D4CD !important;
    }

    /* ------------------------------
       TABLE
    ------------------------------ */

    [data-testid="stDataFrame"] {
        border-radius: 16px;
        overflow: hidden;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATENBANK
# ============================================================

def init_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS diary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entry_date TEXT NOT NULL,
            skin_score INTEGER NOT NULL,
            dryness INTEGER NOT NULL,
            redness INTEGER NOT NULL,
            impurities INTEGER NOT NULL,
            note TEXT
        )
        """
    )

    conn.commit()
    conn.close()


def save_diary_entry(
    entry_date,
    skin_score,
    dryness,
    redness,
    impurities,
    note,
):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO diary
        (entry_date, skin_score, dryness, redness, impurities, note)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            str(entry_date),
            skin_score,
            dryness,
            redness,
            impurities,
            note,
        ),
    )

    conn.commit()
    conn.close()


def load_diary_entries():
    conn = sqlite3.connect(DB_FILE)

    rows = conn.execute(
        """
        SELECT
            id,
            entry_date,
            skin_score,
            dryness,
            redness,
            impurities,
            note
        FROM diary
        ORDER BY entry_date DESC
        """
    ).fetchall()

    conn.close()

    return rows


def delete_diary_entry(entry_id):
    conn = sqlite3.connect(DB_FILE)
    conn.execute(
        "DELETE FROM diary WHERE id = ?",
        (entry_id,),
    )
    conn.commit()
    conn.close()


init_database()


# ============================================================
# SESSION STATE
# ============================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "result" not in st.session_state:
    st.session_state.result = None

if "reset_count" not in st.session_state:
    st.session_state.reset_count = 0

if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = 0

if "quiz_question" not in st.session_state:
    st.session_state.quiz_question = 0


# ============================================================
# KI-MODELL
# ============================================================

@st.cache_resource(
    show_spinner="KI-Modell wird geladen. Der erste Start kann etwas dauern ..."
)
def load_model():
    from transformers import (
        ViTImageProcessor,
        AutoModelForImageClassification,
    )

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model


def classify(image):
    import torch

    processor, model = load_model()

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    probabilities = torch.softmax(
        logits,
        dim=-1,
    )[0]

    results = []

    for index, score in enumerate(
        probabilities.tolist()
    ):
        original = model.config.id2label[index]

        key = (
            original
            .replace("_", " ")
            .replace("-", " ")
            .strip()
            .lower()
        )

        german_name = LABEL_MAP.get(
            key,
            original,
        )

        results.append(
            (
                german_name,
                original,
                float(score),
            )
        )

    return sorted(
        results,
        key=lambda x: x[2],
        reverse=True,
    )


# ============================================================
# BILD-FUNKTIONEN
# ============================================================

def load_valid_image(file):
    data = file.getvalue()

    if len(data) > MAX_FILE_MB * 1024 * 1024:
        return (
            None,
            f"Die Datei ist größer als {MAX_FILE_MB} MB.",
        )

    try:
        Image.open(
            io.BytesIO(data)
        ).verify()

        image = Image.open(
            io.BytesIO(data)
        )

        image = ImageOps.exif_transpose(
            image
        ).convert("RGB")

        return image, None

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ):
        return (
            None,
            "Diese Datei konnte nicht als Bild gelesen werden.",
        )


def check_photo_quality(image):
    gray = np.asarray(
        image.convert("L").resize(
            (256, 256)
        ),
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
            "Das Foto ist ziemlich dunkel. Tageslicht wäre besser."
        )

    if brightness > 200:
        hints.append(
            "Das Foto ist sehr hell. Vermeide grelles Licht."
        )

    if sharpness < 30:
        hints.append(
            "Das Foto wirkt unscharf. Halte die Kamera ruhiger."
        )

    return hints


def reset_image():
    st.session_state.reset_count += 1
    st.session_state.result = None


def build_report(results):
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
        "",
        DISCLAIMER,
    ]

    return "\n".join(lines)


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        padding: 0.7rem 0 1.3rem 0;
        text-align: center;
    ">
        <div style="
            font-size: 2rem;
        ">🌸</div>

        <div style="
            font-size: 1.25rem;
            font-weight: 700;
            color: #4A3328;
        ">
            SkinCareApp
        </div>

        <div style="
            color: #967D72;
            font-size: 0.75rem;
        ">
            Deine persönliche Skincare-Welt
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Startseite",
        "🤖 KI-Hautanalyse",
        "🧴 Persönlicher Pflegeplan",
        "📔 Pflege-Tagebuch",
        "📊 Haut-Statistik",
        "🧪 Wirkstoff-Lexikon",
        "🎯 Hautpflege-Quiz",
        "🔎 Inhaltsstoffe-Check",
        "ℹ️ Info & Datenschutz",
    ],
)


st.sidebar.markdown(
    """
    <div style="
        margin-top: 2rem;
        padding: 1rem;
        background: rgba(255,255,255,0.55);
        border-radius: 18px;
        border: 1px solid rgba(130,90,80,0.08);
    ">
        <div style="
            font-weight: 600;
            color: #4A3328;
        ">
            ✦ Dein SkinCare-Bereich
        </div>

        <div style="
            margin-top: 0.45rem;
            color: #927A70;
            font-size: 0.78rem;
            line-height: 1.6;
        ">
            Analysieren · Planen · Tracken · Lernen
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# STARTSEITE
# ============================================================

def page_home():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                Hallo! 🌸
            </div>

            <div class="hero-subtitle">
                Willkommen in deiner persönlichen SkinCare-Welt.
                Analysiere sichtbare Hautmerkmale, erstelle deine Routine
                und beobachte deine Hautentwicklung.
            </div>

            <div style="margin-top: 1rem;">
                <span class="badge">KI-Analyse</span>
                <span class="badge badge-green">Pflegeplan</span>
                <span class="badge badge-blue">Tagebuch</span>
                <span class="badge badge-purple">Statistik</span>
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Deine SkinCare-Welt")

    cols = st.columns(4)

    features = [
        (
            "🤖",
            "KI-Hautanalyse",
            "Selfie hochladen und sichtbare Merkmale analysieren.",
        ),
        (
            "🧴",
            "Pflegeplan",
            "Deine persönliche Morgen- und Abendroutine.",
        ),
        (
            "📔",
            "Tagebuch",
            "Hautzustand und Veränderungen dokumentieren.",
        ),
        (
            "📊",
            "Statistik",
            "Deine Entwicklung übersichtlich als Diagramm.",
        ),
        (
            "🧪",
            "Wirkstoff-Lexikon",
            "Wichtige Wirkstoffe einfach erklärt.",
        ),
        (
            "🎯",
            "Hautpflege-Quiz",
            "Teste dein Wissen rund um Hautpflege.",
        ),
        (
            "🔎",
            "Inhaltsstoffe",
            "INCI-Liste einfach überprüfen.",
        ),
        (
            "🌿",
            "Skincare-Wissen",
            "Allgemeine Tipps für deine Pflegeroutine.",
        ),
    ]

    for index, item in enumerate(features):
        icon, title, description = item

        with cols[index % 4]:
            st.markdown(
                f"""
                <div class="feature-card">

                    <div class="feature-icon">
                        {icon}
                    </div>

                    <div class="feature-title">
                        {title}
                    </div>

                    <div class="feature-text">
                        {description}
                    </div>

                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.25, 1])

    with col1:
        st.markdown(
            """
            <div class="section-card">

                <div style="
                    color:#4A3328;
                    font-size:1.15rem;
                    font-weight:650;
                ">
                    ✨ Dein heutiger Fokus
                </div>

                <div style="
                    color:#927A70;
                    margin-top:0.45rem;
                    line-height:1.6;
                ">
                    Eine gute Skincare-Routine muss nicht kompliziert sein.
                    Kleine, regelmäßige Schritte sind wichtiger als viele Produkte.
                </div>

                <div class="quote" style="margin-top:1rem;">
                    „Weniger, aber bewusst – das ist die Idee hinter deiner Routine.“
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">

                <div style="
                    color:#4A3328;
                    font-size:1.15rem;
                    font-weight:650;
                ">
                    🌸 App-Übersicht
                </div>

                <div style="
                    color:#927A70;
                    line-height:2;
                    margin-top:0.5rem;
                ">
                    🏠 Startseite<br>
                    🤖 KI-Hautanalyse<br>
                    🧴 Pflegeplan<br>
                    📔 Tagebuch<br>
                    📊 Statistik<br>
                    🧪 Wirkstoff-Lexikon<br>
                    🎯 Quiz
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Tipp: Starte mit der KI-Hautanalyse oder lege direkt "
        "deinen persönlichen Pflegeplan an."
    )


# ============================================================
# KI-HAUTANALYSE
# ============================================================

def page_analysis():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🤖 KI-Hautanalyse
            </div>

            <div class="hero-subtitle">
                Lade ein Selfie hoch oder nutze deine Kamera.
                Die KI analysiert sichtbare Hautmerkmale.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">

        <span class="badge">KI-Analyse</span>
        <span class="badge badge-green">Privatsphäre</span>

        <p style="margin-top:0.8rem;">
        Dein Foto wird innerhalb dieser App verarbeitet und nicht
        als Teil des Analyseverlaufs gespeichert.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    n = st.session_state.reset_count

    source = st.radio(
        "Bildquelle",
        [
            "📁 Foto hochladen",
            "📷 Selfie aufnehmen",
        ],
        horizontal=True,
        label_visibility="collapsed",
    )

    if source == "📁 Foto hochladen":

        file = st.file_uploader(
            "Selfie hochladen",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            key=f"upload_{n}",
        )

    else:

        st.caption(
            "Dein Browser fragt eventuell nach Kamera-Berechtigungen."
        )

        file = st.camera_input(
            "Selfie aufnehmen",
            key=f"camera_{n}",
        )

    if file is None:
        st.markdown(
            """
            <div class="section-card" style="
                text-align:center;
                padding:2.5rem;
            ">

                <div style="font-size:3rem;">
                    📸
                </div>

                <div style="
                    font-weight:650;
                    color:#4A3328;
                ">
                    Bereit für deine Analyse?
                </div>

                <div style="
                    color:#927A70;
                    margin-top:0.4rem;
                ">
                    Lade ein Foto hoch oder öffne deine Kamera.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    image, error = load_valid_image(file)

    if error:
        st.error(error)
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Dein Foto",
            use_container_width=True,
        )

    with col2:
        st.markdown(
            """
            <div class="section-card">

                <div style="
                    font-weight:650;
                    color:#4A3328;
                    font-size:1.1rem;
                ">
                    Foto-Qualität
                </div>

                <p class="small">
                    Für bessere Ergebnisse sollte dein Gesicht
                    gut sichtbar und ausreichend beleuchtet sein.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )

        for hint in check_photo_quality(image):
            st.warning(hint)

        st.success("Foto erfolgreich geladen.")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button(
            "Bild entfernen",
            use_container_width=True,
        ):
            reset_image()
            st.rerun()

    with col2:
        analyze = st.button(
            "✨ Haut analysieren",
            use_container_width=True,
        )

    if analyze:

        try:

            with st.spinner(
                "Deine Haut wird analysiert ..."
            ):

                results = classify(image)

            st.session_state.result = results

            history = st.session_state.history

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
                "Das KI-Modell konnte nicht geladen oder "
                "ausgeführt werden."
            )

            with st.expander(
                "Technische Details"
            ):
                st.code(
                    f"{type(error_info).__name__}: "
                    f"{error_info}"
                )

    if st.session_state.result:
        show_results(
            st.session_state.result
        )


def show_results(results):

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title" style="font-size:2rem;">
                Analyse abgeschlossen ✨
            </div>

            <div class="hero-subtitle">
                Hier siehst du die wichtigsten Ergebnisse deiner Analyse.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    top_name, _, top_score = results[0]

    if top_score < CONFIDENCE_THRESHOLD:

        st.warning(
            "Das Modell konnte dieses Bild nicht sicher zuordnen."
        )

    else:

        explanation, tips = CARE.get(
            top_name,
            (
                "",
                [],
            ),
        )

        st.markdown(
            f"""
            <div class="section-card">

                <span class="badge badge-green">
                    Höchster Modellwert
                </span>

                <h3 style="margin-top:0.7rem;">
                    {top_name}
                </h3>

                <p>
                    {explanation}
                </p>

                <div style="
                    font-size:1.8rem;
                    font-weight:700;
                    color:#4A3328;
                ">
                    {top_score:.0%}
                </div>

                <div class="small">
                    Modell-Sicherheit
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        if tips:

            st.markdown("### 💡 Allgemeine Pflegetipps")

            for tip in tips:
                st.markdown(
                    f"""
                    <div class="routine-step">
                        🌿 {tip}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("### 📊 Alle Modellwerte")

    for name, _, score in results:

        st.progress(
            min(
                max(
                    score,
                    0.0,
                ),
                1.0,
            ),
            text=f"{name} · {score:.0%}",
        )

    st.caption(
        "Das Modell wählt eine Kategorie pro Bild. "
        "Die Werte sind keine medizinischen Messwerte."
    )

    st.download_button(
        "📄 Analysebericht herunterladen",
        build_report(results),
        file_name="skincare_bericht.txt",
        mime="text/plain",
    )


# ============================================================
# PFLEGEPLAN
# ============================================================

def page_routine():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🧴 Persönlicher Pflegeplan
            </div>

            <div class="hero-subtitle">
                Erstelle eine einfache Morgen- und Abendroutine,
                die du Schritt für Schritt verfolgen kannst.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    skin_type = st.selectbox(
        "Welcher Hauttyp passt am ehesten zu dir?",
        [
            "Noch nicht sicher",
            "Normal",
            "Trocken",
            "Ölig",
            "Mischhaut",
            "Empfindlich",
        ],
    )

    focus = st.selectbox(
        "Was möchtest du besonders berücksichtigen?",
        [
            "Ausgeglichene Pflege",
            "Feuchtigkeit",
            "Unreinheiten",
            "Rötungen",
            "Hautbarriere",
            "Sonnenschutz",
        ],
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="section-card">

                <span class="badge">
                    ☀️ MORGEN
                </span>

                <h3>
                    Deine Morgenroutine
                </h3>

            </div>
            """,
            unsafe_allow_html=True,
        )

        morning = [
            (
                "Sanfte Reinigung",
                "Entferne Schweiß und überschüssiges Hautfett.",
            ),
            (
                "Feuchtigkeit",
                "Creme passend zu deinem Hautgefühl.",
            ),
            (
                "Sonnenschutz",
                "Der wichtigste Schritt am Morgen.",
            ),
        ]

        for title, description in morning:

            st.checkbox(
                title,
                key=f"morning_{title}",
            )

            st.markdown(
                f"""
                <div class="routine-desc"
                     style="margin:-0.5rem 0 0.8rem 1.9rem;">
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col2:

        st.markdown(
            """
            <div class="section-card">

                <span class="badge badge-purple">
                    🌙 ABEND
                </span>

                <h3>
                    Deine Abendroutine
                </h3>

            </div>
            """,
            unsafe_allow_html=True,
        )

        evening = [
            (
                "Reinigung",
                "Entferne Make-up, Schmutz und Sonnenschutz.",
            ),
            (
                "Wirkstoff",
                "Nur wenn er zu deiner Routine passt.",
            ),
            (
                "Feuchtigkeitspflege",
                "Unterstütze deine Haut über Nacht.",
            ),
        ]

        for title, description in evening:

            st.checkbox(
                title,
                key=f"evening_{title}",
            )

            st.markdown(
                f"""
                <div class="routine-desc"
                     style="margin:-0.5rem 0 0.8rem 1.9rem;">
                    {description}
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="quote">
            <b>Dein Fokus:</b> {focus}<br>
            <span style="font-size:0.9rem;">
                Hauttyp: {skin_type}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "Die Routine ist eine allgemeine Orientierung und "
        "keine individuelle medizinische Behandlung."
    )


# ============================================================
# PFLEGE-TAGEBUCH
# ============================================================

def page_diary():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                📔 Pflege-Tagebuch
            </div>

            <div class="hero-subtitle">
                Halte täglich fest, wie sich deine Haut anfühlt
                und entdecke langfristige Veränderungen.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:

        entry_date = st.date_input(
            "Datum",
            value=date.today(),
        )

        skin_score = st.slider(
            "Wie zufrieden bist du heute mit deiner Haut?",
            1,
            5,
            3,
        )

        dryness = st.slider(
            "Trockenheit",
            1,
            5,
            1,
        )

    with col2:

        redness = st.slider(
            "Rötungen",
            1,
            5,
            1,
        )

        impurities = st.slider(
            "Unreinheiten",
            1,
            5,
            1,
        )

        note = st.text_area(
            "Notiz",
            placeholder=(
                "Wie fühlt sich deine Haut heute an?"
            ),
            height=130,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "💾 Tagebucheintrag speichern",
        use_container_width=True,
    ):

        save_diary_entry(
            entry_date,
            skin_score,
            dryness,
            redness,
            impurities,
            note,
        )

        st.success(
            "Dein Tagebucheintrag wurde gespeichert."
        )

    st.markdown("### Deine letzten Einträge")

    rows = load_diary_entries()

    if not rows:

        st.info(
            "Noch keine Einträge vorhanden."
        )

        return

    for row in rows[:10]:

        entry_id = row[0]
        entry_date = row[1]
        skin_score = row[2]
        dryness = row[3]
        redness = row[4]
        impurities = row[5]
        note = row[6]

        with st.expander(
            f"📅 {entry_date} · Hautgefühl {skin_score}/5"
        ):

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Trockenheit",
                f"{dryness}/5",
            )

            c2.metric(
                "Rötungen",
                f"{redness}/5",
            )

            c3.metric(
                "Unreinheiten",
                f"{impurities}/5",
            )

            if note:
                st.write(note)

            if st.button(
                "Eintrag löschen",
                key=f"delete_{entry_id}",
            ):

                delete_diary_entry(
                    entry_id
                )

                st.rerun()


# ============================================================
# HAUT-STATISTIK
# ============================================================

def page_statistics():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                📊 Haut-Statistik
            </div>

            <div class="hero-subtitle">
                Dein persönlicher Überblick über deine Tagebuchwerte.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    rows = load_diary_entries()

    if not rows:

        st.info(
            "Für deine Statistik brauchst du zuerst "
            "einige Tagebucheinträge."
        )

        st.markdown(
            """
            <div class="section-card" style="text-align:center;">

                <div style="font-size:3rem;">
                    📈
                </div>

                <div style="
                    font-weight:650;
                    color:#4A3328;
                ">
                    Deine Statistik entsteht hier
                </div>

                <div class="small">
                    Je mehr Einträge du sammelst,
                    desto interessanter wird dein Verlauf.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    import pandas as pd

    df = pd.DataFrame(
        rows,
        columns=[
            "id",
            "Datum",
            "Hautbewertung",
            "Trockenheit",
            "Rötungen",
            "Unreinheiten",
            "Notiz",
        ],
    )

    df["Datum"] = pd.to_datetime(
        df["Datum"]
    )

    df = df.sort_values(
        "Datum"
    )

    latest = df.iloc[-1]

    avg_score = df[
        "Hautbewertung"
    ].mean()

    entries = len(df)

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-number">
                    {avg_score:.1f}/5
                </div>

                <div class="stat-label">
                    Durchschnittliche Hautbewertung
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-number">
                    {entries}
                </div>

                <div class="stat-label">
                    Tagebucheinträge
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:

        st.markdown(
            f"""
            <div class="stat-card">

                <div class="stat-number">
                    {latest["Hautbewertung"]}/5
                </div>

                <div class="stat-label">
                    Letzte Bewertung
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">

        <h3>
            📈 Entwicklung deiner Hautbewertung
        </h3>

        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_df = df[
        [
            "Datum",
            "Hautbewertung",
        ]
    ].set_index(
        "Datum"
    )

    st.line_chart(
        chart_df
    )

    st.markdown("### 🔎 Weitere Werte")

    other_df = df[
        [
            "Datum",
            "Trockenheit",
            "Rötungen",
            "Unreinheiten",
        ]
    ].set_index(
        "Datum"
    )

    st.line_chart(
        other_df
    )

    st.caption(
        "Die Statistik basiert ausschließlich auf deinen eigenen Tagebucheinträgen."
    )


# ============================================================
# WIRKSTOFF-LEXIKON
# ============================================================

def page_actives():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🧪 Wirkstoff-Lexikon
            </div>

            <div class="hero-subtitle">
                Entdecke wichtige Wirkstoffe der Hautpflege
                und erfahre, wofür sie eingesetzt werden.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    search = st.text_input(
        "Wirkstoff suchen",
        placeholder="z. B. Niacinamid",
    )

    selected = []

    for name, data in ACTIVES.items():

        if (
            not search.strip()
            or search.lower()
            in name.lower()
            or search.lower()
            in data["category"].lower()
        ):
            selected.append(
                (
                    name,
                    data,
                )
            )

    if not selected:

        st.info(
            "Keinen passenden Wirkstoff gefunden."
        )

        return

    for name, data in selected:

        with st.expander(
            f"{data['icon']} {name} · {data['category']}"
        ):

            st.markdown(
                f"""
                <div class="section-card">

                    <span class="badge">
                        {data["category"]}
                    </span>

                    <p style="margin-top:0.8rem;">
                        {data["description"]}
                    </p>

                    <h4>
                        Typische Eigenschaften
                    </h4>

                </div>
                """,
                unsafe_allow_html=True,
            )

            for benefit in data["benefits"]:
                st.markdown(
                    f"✓ {benefit}"
                )

            st.info(
                f"💡 {data['tip']}"
            )


# ============================================================
# HAUTPFLEGE-QUIZ
# ============================================================

def page_quiz():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🎯 Hautpflege-Quiz
            </div>

            <div class="hero-subtitle">
                Teste dein Wissen über Hautpflege und Wirkstoffe.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    questions = [
        {
            "question": "Welcher Schritt gehört morgens besonders zur Basisroutine?",
            "options": [
                "Sonnenschutz",
                "Sehr starkes Peeling",
                "Parfüm auf die Haut",
                "Gesicht mit heißem Wasser waschen",
            ],
            "answer": "Sonnenschutz",
            "explanation": (
                "Sonnenschutz ist ein wichtiger Bestandteil einer "
                "allgemeinen Morgenroutine."
            ),
        },

        {
            "question": "Welcher Wirkstoff wird häufig mit Feuchtigkeitspflege verbunden?",
            "options": [
                "Hyaluronsäure",
                "Parfum",
                "Menthol",
                "Alkohol Denat.",
            ],
            "answer": "Hyaluronsäure",
            "explanation": (
                "Hyaluronsäure wird häufig als feuchtigkeitsbindender "
                "Wirkstoff eingesetzt."
            ),
        },

        {
            "question": "Was ist bei empfindlicher Haut grundsätzlich sinnvoll?",
            "options": [
                "Milde Pflege verwenden",
                "Jeden Tag stark peelen",
                "Sehr heißes Wasser benutzen",
                "Möglichst viele Produkte gleichzeitig verwenden",
            ],
            "answer": "Milde Pflege verwenden",
            "explanation": (
                "Eine einfache und milde Routine kann helfen, "
                "unnötige Reizung zu vermeiden."
            ),
        },

        {
            "question": "Welcher Wirkstoff gehört zur Vitamin-A-Gruppe?",
            "options": [
                "Retinol",
                "Glycerin",
                "Hyaluronsäure",
                "Panthenol",
            ],
            "answer": "Retinol",
            "explanation": (
                "Retinol gehört zu den Vitamin-A-Derivaten "
                "und wird häufig in Anti-Aging-Pflege eingesetzt."
            ),
        },

        {
            "question": "Was sollte man mit einem schmerzhaften Hautproblem tun?",
            "options": [
                "Bei Bedarf ärztlich abklären lassen",
                "Immer selbst ausdrücken",
                "Sehr stark schrubben",
                "Problem ignorieren",
            ],
            "answer": "Bei Bedarf ärztlich abklären lassen",
            "explanation": (
                "Starke oder schmerzhafte Hautprobleme sollten "
                "gegebenenfalls professionell abgeklärt werden."
            ),
        },
    ]

    total = len(questions)

    if st.session_state.quiz_question >= total:

        st.markdown(
            f"""
            <div class="section-card" style="text-align:center;">

                <div style="font-size:3rem;">
                    🏆
                </div>

                <div style="
                    font-size:1.7rem;
                    font-weight:700;
                    color:#4A3328;
                ">
                    Quiz abgeschlossen!
                </div>

                <div style="
                    font-size:1.1rem;
                    color:#8D756A;
                    margin-top:0.5rem;
                ">
                    Dein Ergebnis:
                    <b>
                        {st.session_state.quiz_score}
                        / {total}
                    </b>
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button(
            "Quiz neu starten",
            use_container_width=True,
        ):

            st.session_state.quiz_question = 0
            st.session_state.quiz_score = 0
            st.rerun()

        return

    current = questions[
        st.session_state.quiz_question
    ]

    progress = (
        st.session_state.quiz_question
        / total
    )

    st.progress(
        progress,
        text=(
            f"Frage "
            f"{st.session_state.quiz_question + 1}"
            f" von {total}"
        ),
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="section-card">

            <span class="badge">
                Frage {st.session_state.quiz_question + 1}
            </span>

            <h3 style="margin-top:0.8rem;">
                {current["question"]}
            </h3>

        </div>
        """,
        unsafe_allow_html=True,
    )

    answer = st.radio(
        "Antwort auswählen",
        current["options"],
        key=f"quiz_{st.session_state.quiz_question}",
        label_visibility="collapsed",
    )

    if st.button(
        "Antwort prüfen",
        use_container_width=True,
    ):

        if answer == current["answer"]:

            st.session_state.quiz_score += 1

            st.success(
                f"Richtig! {current['explanation']}"
            )

        else:

            st.error(
                f"Nicht ganz. Die richtige Antwort ist: "
                f"{current['answer']}. "
                f"{current['explanation']}"
            )

        st.session_state.quiz_question += 1

        st.button(
            "Weiter",
            key=f"next_{st.session_state.quiz_question}",
        )

        st.rerun()


# ============================================================
# INHALTSSTOFFE
# ============================================================

def page_ingredients():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                🔎 Inhaltsstoffe-Check
            </div>

            <div class="hero-subtitle">
                Füge eine INCI-Liste ein und finde bekannte
                Wirkstoffe und ausgewählte Hinweise.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "Zutatenliste",
        height=180,
        placeholder=(
            "Aqua, Glycerin, Niacinamide, "
            "Parfum, Panthenol ..."
        ),
    )

    if not text.strip():

        st.markdown(
            """
            <div class="section-card" style="text-align:center;">

                <div style="font-size:2.5rem;">
                    🧪
                </div>

                <div style="
                    font-weight:650;
                    color:#4A3328;
                ">
                    Zutatenliste hier einfügen
                </div>

                <div class="small">
                    Die App sucht anschließend nach bekannten Begriffen.
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        return

    lowered = text.lower()

    found = []

    for keyword, (
        icon,
        note,
    ) in INGREDIENTS.items():

        if keyword in lowered:

            found.append(
                (
                    keyword,
                    icon,
                    note,
                )
            )

    st.markdown("### Ergebnis")

    if not found:

        st.info(
            "Keine Stoffe aus unserer kleinen Stichwortliste gefunden."
        )

        return

    for keyword, icon, note in found:

        st.markdown(
            f"""
            <div class="routine-step">

                <div style="
                    font-weight:650;
                    color:#4A3328;
                ">
                    {icon} {keyword.title()}
                </div>

                <div class="routine-desc">
                    {note}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        "Der Check ist nur eine einfache Stichwortsuche "
        "und keine medizinische oder vollständige Produktbewertung."
    )


# ============================================================
# INFO
# ============================================================

def page_about():

    st.markdown(
        """
        <div class="hero">

            <div class="hero-title">
                ℹ️ Über die SkinCareApp
            </div>

            <div class="hero-subtitle">
                Informationen über Projekt, KI-Modell und Datenschutz.
            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">

        <h3>
            🌸 Unser Projekt
        </h3>

        <p>
            Die SkinCareApp ist ein Schulprojekt rund um Hautpflege,
            künstliche Intelligenz und digitale Gesundheitsbildung.
        </p>

        <p>
            Die App verbindet eine KI-Hautanalyse mit praktischen
            Funktionen wie Pflegeplan, Tagebuch, Statistik,
            Wirkstoff-Lexikon und Quiz.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">

        <h3>
            🤖 KI-Modell
        </h3>

        <p>
            Verwendet wird das vortrainierte Hugging-Face-Modell:
        </p>

        <span class="badge">
            varun1505/face-characteristics
        </span>

        <p class="small" style="margin-top:0.8rem;">
            Das Modell ordnet ein Bild einer sichtbaren
            Hautmerkmal-Kategorie zu. Die Ergebnisse können falsch sein.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="section-card">

        <h3>
            🔒 Datenschutz
        </h3>

        <p>
            Die hochgeladenen Fotos werden für die Analyse verarbeitet
            und nicht als Teil des Analyseverlaufs gespeichert.
        </p>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.warning(DISCLAIMER)


# ============================================================
# SEITEN ROUTER
# ============================================================

if page == "🏠 Startseite":
    page_home()

elif page == "🤖 KI-Hautanalyse":
    page_analysis()

elif page == "🧴 Persönlicher Pflegeplan":
    page_routine()

elif page == "📔 Pflege-Tagebuch":
    page_diary()

elif page == "📊 Haut-Statistik":
    page_statistics()

elif page == "🧪 Wirkstoff-Lexikon":
    page_actives()

elif page == "🎯 Hautpflege-Quiz":
    page_quiz()

elif page == "🔎 Inhaltsstoffe-Check":
    page_ingredients()

elif page == "ℹ️ Info & Datenschutz":
    page_about()


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:3rem;
        padding:1.2rem;
        text-align:center;
        color:#9B8278;
        font-size:0.75rem;
    ">
        🌸 SkinCareApp · Schulprojekt ·
        KI-gestützte Hautpflege-Lernapp
    </div>
    """,
    unsafe_allow_html=True,
)
