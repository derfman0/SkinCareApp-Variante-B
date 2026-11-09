# SkinCareApp (Variante 2: Hugging Face)

Minimalistische Streamlit-App, die sichtbare Hautmerkmale auf einem Selfie mit einem
vortrainierten Hugging-Face-Modell einordnet. Schulprojekt, keine medizinische Diagnose.

## Funktionen
- Selfie hochladen oder mit der Kamera aufnehmen
- Foto-Qualitätscheck (Helligkeit, Schärfe)
- KI-Analyse mit Modellwerten, Pflegetipps und Bericht zum Download
- Verlauf der Sitzung (ohne Fotos)
- Hauttyp-Quiz
- Inhaltsstoffe-Check (einfache Stichwortliste)

## Modell
`varun1505/face-characteristics` (Vision Transformer, ca. 86 Mio. Parameter).
Klassen laut Modellseite: acne, eyes dark circles, puffy eyes, skin pigmentation,
skin redness, skin texture, wrinkled face.
Selbst angegebene Genauigkeit: 57 %. Lizenz auf der Modellseite nicht angegeben (vor Abgabe prüfen).
Nicht erkannt: Poren, trockene Haut.

## Lokal starten
```
pip install -r requirements.txt
streamlit run app.py
```
Beim ersten Start wird das Modell (ca. 350 MB) heruntergeladen.

## Datenschutz
Das Modell läuft in der App selbst. Selfies werden nur im Arbeitsspeicher verarbeitet,
nicht gespeichert und nicht an einen KI-Dienst gesendet. Kein API-Token nötig.

## Deployment
GitHub-Repository -> share.streamlit.io -> Hauptdatei app.py -> Deploy.
