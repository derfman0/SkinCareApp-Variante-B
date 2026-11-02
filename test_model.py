import tensorflow as tf
from huggingface_hub import hf_hub_download

repo = "SopanhaZinII/facial-condition-model"
print("Lade Modell …")
path = hf_hub_download(repo_id=repo, filename="finetuned_best.h5")
model = tf.keras.models.load_model(path, compile=False)
print("Modell geladen!")
print("Eingabeform:", model.input_shape)
print("Ausgabeform:", model.output_shape)
