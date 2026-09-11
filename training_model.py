import json
import numpy as np
from model import get_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from helpers import get_word_ids, get_sequences_and_labels, load_lsa_json
from constants import *

# Ruta por defecto del JSON; si lo tenés en otro lado, cambiá aquí o pásalo por argv si querés
LSA_JSON_PATH = "lsa_samples.json"

def training_model(model_path, epochs=500):
    # 1) Cargar orden de clases
    word_ids = get_word_ids(WORDS_JSON_PATH)
    print("[INFO] word_ids inicial:", word_ids)

    # 2) Solo leo H5 de las palabras que realmente tienen archivo
    labels_with_h5 = [w for w in word_ids
                      if os.path.exists(os.path.join(KEYPOINTS_PATH, f"{w}.h5"))]
    if len(labels_with_h5) < len(word_ids):
        faltan = sorted(set(word_ids) - set(labels_with_h5))
        print("[INFO] Ignorando H5 inexistentes:", faltan)

    # get_sequences_and_labels devuelve índices relativos a 'labels_with_h5'
    sequences_h5, labels_idx_h5_local = get_sequences_and_labels(labels_with_h5)
    labels_txt_h5 = [labels_with_h5[i] for i in labels_idx_h5_local]
    print(f"[INFO] H5: {len(sequences_h5)} secuencias")

    # 3) Cargar JSON (puede traer nuevas etiquetas)
    try:
        sequences_json, labels_txt_json = load_lsa_json("lsa_samples.json")
        print(f"[INFO] JSON: {len(sequences_json)} secuencias; clases={set(labels_txt_json)}")
    except FileNotFoundError:
        sequences_json, labels_txt_json = [], []
        print("[WARN] No se encontró lsa_samples.json; entreno solo con H5.")

    # 4) Si el JSON trae etiquetas nuevas, las agrego a words.json (al final)
    if labels_txt_json:
        new_labels = [lab for lab in sorted(set(labels_txt_json)) if lab not in word_ids]
        if new_labels:
            print("[INFO] Agregando nuevas etiquetas a words.json:", new_labels)
            word_ids = word_ids + new_labels
            with open(WORDS_JSON_PATH, "r", encoding="utf-8") as f:
                wdata = json.load(f)
            wdata["word_ids"] = word_ids
            with open(WORDS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(wdata, f, ensure_ascii=False, indent=2)
            print("[OK] words.json actualizado:", word_ids)

    # 5) Mapear ambos (H5 y JSON) a índices globales según word_ids
    word_to_idx = {w: i for i, w in enumerate(word_ids)}
    labels_idx_h5 = [word_to_idx[w] for w in labels_txt_h5]
    labels_idx_json = [word_to_idx[w] for w in labels_txt_json] if labels_txt_json else []

    # 6) Fusionar
    sequences_all = sequences_h5 + sequences_json
    labels_idx_all = labels_idx_h5 + labels_idx_json

    # 7) Padding a longitud fija (usa tu MODEL_FRAMES y padding='pre' como ya haces)
    sequences_all = pad_sequences(
        sequences_all,
        maxlen=int(MODEL_FRAMES),
        padding='pre',
        truncating='post',
        dtype='float16'
    )  # -> (N, T, D)

    X = np.array(sequences_all)                               # (N, T, D)
    y = to_categorical(labels_idx_all, num_classes=len(word_ids)).astype(int)

    # 8) Split + early stopping + train
    early_stopping = EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = get_model(int(MODEL_FRAMES), len(word_ids))
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=epochs, batch_size=8, callbacks=[early_stopping])

    model.summary()
    model.save(model_path)
    print("[OK] Modelo guardado en:", model_path)

if __name__ == "__main__":
    training_model(MODEL_PATH)
