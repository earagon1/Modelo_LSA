"""Exporta un .keras a TFLite sin Flex y deja todo lo necesario para publicarlo (T-A23).

No toca models/: escribe en runs/<fecha>_sin_flex/
  - actions_15_opt.tflite      variante dinámica, solo ops builtin (lsa.export)
  - verificacion.json          contra Keras y contra el .tflite publicado hoy,
                               sobre todas las secuencias de data/keypoints
  - paridad_modelo.json        fixture para el test instrumentado de la app:
                               entradas reales + probabilidades esperadas,
                               para comprobar que el runtime de Android da lo
                               mismo que el de Python

Uso, desde la raíz de modelo_LSA:
    python scripts/exportar_sin_flex.py --modelo models/actions_15.keras \
        --publicado ../../TraductorLSA/app/src/main/assets/actions_15_opt.tflite
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa.export import convertir, inferir_tflite, ops_flex

T, D = 15, 126
POR_CLASE_FIXTURE = 2


def datos():
    """Secuencias v1 tal como entraron al entrenamiento (pad adelante): es la misma
    entrada para el Keras, el .tflite publicado y el nuevo, así se comparan bien."""
    from constants import WORDS_JSON_PATH
    from helpers import get_sequences_and_labels, get_word_ids
    palabras = get_word_ids(WORDS_JSON_PATH)
    secuencias, etiquetas = get_sequences_and_labels(palabras)
    x = np.zeros((len(secuencias), T, D), np.float32)
    for i, s in enumerate(secuencias):
        s = np.asarray(s, np.float32)[:T]
        x[i, T - len(s):] = s
    return x, np.asarray(etiquetas), palabras


def comparar(a, b):
    return {"top1_igual": float((a.argmax(1) == b.argmax(1)).mean()),
            "max_abs": float(np.abs(a - b).max()),
            "media_abs": float(np.abs(a - b).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="models/actions_15.keras")
    ap.add_argument("--publicado", help=".tflite que hoy está en la app, para comparar")
    ap.add_argument("--salida", default=os.path.join("runs", f"{date.today().isoformat()}_sin_flex"))
    args = ap.parse_args()

    from keras.models import load_model
    modelo = load_model(args.modelo)
    x, y, palabras = datos()

    tfl = convertir(modelo, T, D, "dinamica")
    assert not ops_flex(tfl), "quedaron ops Flex"
    p_nuevo = inferir_tflite(tfl, x)
    p_keras = modelo.predict(x, verbose=0, batch_size=256)

    verif = {
        "modelo_keras": args.modelo,
        "n_secuencias": len(x),
        "kb": round(len(tfl) / 1024, 1),
        "sha256": hashlib.sha256(tfl).hexdigest(),
        "ops_flex": ops_flex(tfl),
        "vs_keras": comparar(p_nuevo, p_keras),
        "accuracy_datos_vistos": float((p_nuevo.argmax(1) == y).mean()),
    }
    if args.publicado:
        with open(args.publicado, "rb") as f:
            publicado = f.read()
        p_pub = inferir_tflite(publicado, x)
        verif["vs_publicado"] = comparar(p_nuevo, p_pub)
        verif["publicado_sha256"] = hashlib.sha256(publicado).hexdigest()
        verif["publicado_kb"] = round(len(publicado) / 1024, 1)

    # Fixture para la app: las primeras secuencias de cada clase.
    idx = [i for c in range(len(palabras)) for i in np.where(y == c)[0][:POR_CLASE_FIXTURE]]
    fixture = {
        "modelo_sha256": verif["sha256"],
        "word_ids": palabras,
        "tolerancia": 1e-4,
        "casos": [{"esperado_label": palabras[int(y[i])],
                   "entrada": np.round(x[i], 7).tolist(),
                   "probs": p_nuevo[i].tolist()} for i in idx],
    }
    # Las probabilidades esperadas se recalculan sobre la entrada redondeada que
    # efectivamente queda en el JSON, para que la comparación sea exacta.
    entradas = np.asarray([c["entrada"] for c in fixture["casos"]], np.float32)
    for c, p in zip(fixture["casos"], inferir_tflite(tfl, entradas), strict=True):
        c["probs"] = p.tolist()

    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "actions_15_opt.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(args.salida, "verificacion.json"), "w", encoding="utf-8") as f:
        json.dump(verif, f, indent=2, ensure_ascii=False)
    with open(os.path.join(args.salida, "paridad_modelo.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixture, f, separators=(",", ":"))
    print(json.dumps(verif, indent=2, ensure_ascii=False))
    print(f"\n-> {args.salida}")


if __name__ == "__main__":
    main()
