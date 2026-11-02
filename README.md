# SkinCareApp – Variante B

Streamlit-Schulprojekt mit einem vortrainierten Keras/TensorFlow-Modell von Hugging Face.

## Modell
https://huggingface.co/SopanhaZinII/facial-condition-model

Die Modellkarte nennt fünf Merkmale: Acne, Dry Skin, Oily Skin, Dark Spots und Wrinkles. Die App lädt `finetuned_best.h5` beim ersten Aufruf herunter; die Modelldatei muss deshalb nicht ins GitHub-Repository.

**Wichtige Einschränkung:** Die Modelldokumentation legt Bildnormalisierung und genaue Reihenfolge der Ausgabeklassen nicht ausreichend eindeutig fest. Die App zeigt die Scores daher ausdrücklich experimentell an. Vor einer fachlichen Interpretation müssen Vorverarbeitung und Klassenreihenfolge verifiziert werden. Keine medizinische Diagnose.

## Lokal starten
Python 3.11 empfohlen.

```bash
python -m venv .venv
```
Windows:
```bash
.venv\\Scripts\\activate
```
macOS/Linux:
```bash
source .venv/bin/activate
```
Dann:
```bash
pip install -r requirements.txt
python test_model.py
streamlit run app.py
```

## Streamlit Community Cloud
1. Projekt in ein GitHub-Repository hochladen.
2. https://share.streamlit.io/ öffnen und anmelden.
3. New app wählen, Repository verbinden.
4. Branch `main` und Hauptdatei `app.py` wählen.
5. Deploy starten.

## Sicherheit und Datenschutz
- Keine Fotos dauerhaft speichern oder öffentlich teilen.
- Keine Zugangsdaten in GitHub hochladen.
- `.streamlit/secrets.toml` ist in `.gitignore` ausgeschlossen.
- Für das öffentliche Modell wird im aktuellen Code kein Token benötigt.

## Hinweis zur Aufgabenstellung
Dieses Projekt nutzt `huggingface_hub` + TensorFlow/Keras und nicht die Transformers/PyTorch-Pipeline. Es ist ein vortrainiertes Hugging-Face-Modell, aber falls ausdrücklich `transformers + torch` gefordert sind, muss die Lehrkraft diesen Modellweg freigeben oder ein anderes Modell gewählt werden.
