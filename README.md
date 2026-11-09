# SkinCareApp – Hugging Face edition

Eine Streamlit-Web-App als Schulprojekt. Sie lädt ein vortrainiertes Bildklassifikationsmodell von Hugging Face zur Laufzeit. **Es werden keine Modelldateien in diesem Repository gespeichert.**

## Funktionen
- Bild aus der Galerie hochladen
- Selfie mit der Browser-Kamera aufnehmen
- Bildvorschau und Größen-/Formatprüfung
- Vorhersage und Modell-Softmax-Score anzeigen
- Skincare Guide, Morning/Evening Routine und About & Privacy

## Verwendetes Modell
`LaurianeMD/vit-skin-disease`

Wichtig: Das Modell ist ein dermatologiebezogenes Bildklassifikationsmodell. Es erkennt nicht gezielt die fünf SkinCareApp-Merkmale (Skin Imperfections, Redness, Visible Pores, Dry Skin, Pigmentation). Die Vorhersage ist keine medizinische Diagnose und darf nicht als solche interpretiert werden.

## Lokal starten
Python 3.10 oder 3.11 empfohlen.

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Abhängigkeiten installieren:
```bash
pip install -r requirements.txt
```

App starten:
```bash
streamlit run app.py
```

Beim ersten Start wird das Modell von Hugging Face heruntergeladen. Dafür ist eine Internetverbindung erforderlich. Der Download kann aufgrund der Modellgröße etwas dauern.

## GitHub-Dateien
Die Datei `.streamlit/secrets.toml` ist für lokale Geheimnisse gedacht und wird durch `.gitignore` vom Git-Upload ausgeschlossen. Für das hier verwendete öffentliche Modell ist kein Token notwendig.

Hinweis zur Streamlit-Konfiguration: Streamlit erkennt seine Konfigurationsdatei standardmäßig unter `.streamlit/config.toml`. Die zusätzlich gewünschte Datei `.streamlit_config.toml` liegt als Projektdatei bei; für die tatsächliche Streamlit-Konfiguration wird `.streamlit/config.toml` verwendet.

## Datenschutz
Das Programm speichert Bilder nicht absichtlich in Dateien oder einer Datenbank. Für die Analyse wird das Bild jedoch an den Server übertragen, auf dem die Streamlit-App läuft. Bei einer Veröffentlichung muss deshalb die Hosting- und Datenschutzsituation berücksichtigt werden. Die App ruft in dieser Version nicht die gehostete Hugging-Face-Inference-API auf; sie lädt das Modell und führt die Inferenz auf dem App-Server aus.
