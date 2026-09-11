# verify_dataset_mix.py
import os, json
from collections import Counter
from helpers import get_word_ids, get_sequences_and_labels, load_lsa_json
from constants import WORDS_JSON_PATH, KEYPOINTS_PATH

LSA_JSON_PATH = "lsa_samples.json"  # ajustá si está en otro lado

# 1) Orden oficial (clases del modelo)
word_ids = get_word_ids(WORDS_JSON_PATH)

# 2) De ese orden, quedate solo con las que tienen .h5 presente
labels_with_h5 = [w for w in word_ids if os.path.exists(os.path.join(KEYPOINTS_PATH, f"{w}.h5"))]
missing_h5 = sorted(set(word_ids) - set(labels_with_h5))

# 3) Cargar H5 solo para las que existen
seq_h5, lab_idx_h5 = get_sequences_and_labels(labels_with_h5)
labels_h5 = [labels_with_h5[i] for i in lab_idx_h5]

# 4) Cargar JSON (si no existe, sigue vacío)
try:
    seq_json, lab_txt_json = load_lsa_json(LSA_JSON_PATH)
except FileNotFoundError:
    seq_json, lab_txt_json = [], []

print("Clases (orden del modelo):", word_ids)
print("Faltan estos .h5 (esperable con Opción A):", missing_h5)
print("H5 counts:", Counter(labels_h5))
print("JSON counts:", Counter(lab_txt_json))

# Ayuda visual: ¿hay muestras JSON para palabras sin H5?
json_only = sorted(set(lab_txt_json) - set(labels_with_h5))
print("Sólo en JSON (sin H5):", json_only)
