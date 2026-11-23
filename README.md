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
- Quelle: https://huggingface.co/Muqtadar08/skin_type_classifier
- Lizenz: Apache-2.0 (laut Modellseite)
- Aufgabe: ordnet ein Gesichtsfoto einer von drei Kategorien zu: trockene, normale oder fettige Haut
- Genauigkeit: 61,61 % auf einem Testdatensatz mit 672 Bildern (Angabe der Modellseite)
- Nicht erkannt: Unreinheiten, Rötungen, Pigmentierung, Poren
  
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
- Quelle: https://huggingface.co/Muqtadar08/skin_type_classifier
- Lizenz: Apache-2.0 (laut Modellseite)
- Aufgabe: ordnet ein Gesichtsfoto einer von drei Kategorien zu: trockene, normale oder fettige Haut
- Genauigkeit: 61,61 % auf einem Testdatensatz mit 672 Bildern (Angabe der Modellseite)
- Nicht erkannt: Unreinheiten, Rötungen, Pigmentierung, Poren
