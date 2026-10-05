"""Prueba del export sin Flex (PY-X1). No toca models/ ni escribe .tflite.

Convierte a f32, dinámica e int8 solo con ops builtin (`lsa.export.convertir`),
verifica que no quede ningún op Flex y compara top-1 y probabilidades contra Keras.

- Sin `--modelo`: arquitectura de producción con pesos aleatorios y entradas
  aleatorias. Prueba que la conversión es posible, pero **no** que sea fiel: un
  modelo aleatorio esconde errores numéricos grandes (ver el docstring de
  lsa/export.py).
- Con `--modelo models/actions_15.keras`: modelo entrenado sobre las secuencias
  reales de data/keypoints (solo lectura). Esta es la prueba que vale.

Uso, desde la raíz de modelo_LSA:
    python scripts/probar_export_sin_flex.py --modelo models/actions_15.keras --json salida.json
"""
import argparse
import json
import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa.export import convertir, inferir_tflite, ops_del_modelo, ops_flex

T, D, C = 15, 126, 12


def datos_reales():
    """Secuencias de los .h5 con el preprocesamiento v1 de training_model (pad adelante)."""
    from constants import WORDS_JSON_PATH
    from helpers import get_sequences_and_labels, get_word_ids
    secuencias, etiquetas = get_sequences_and_labels(get_word_ids(WORDS_JSON_PATH))
    x = np.zeros((len(secuencias), T, D), np.float32)
    for i, s in enumerate(secuencias):
        s = np.asarray(s, np.float32)[:T]
        x[i, T - len(s):] = s
    return x, np.asarray(etiquetas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", help=".keras entrenado (si no, pesos aleatorios)")
    ap.add_argument("--json", help="dónde guardar los resultados")
    args = ap.parse_args()

    tf.keras.utils.set_random_seed(0)
    if args.modelo:
        from keras.models import load_model
        modelo = load_model(args.modelo)
        x, y = datos_reales()
    else:
        from model import get_model
        modelo = get_model(T, C)
        x, y = np.random.default_rng(0).uniform(-0.1, 1.0, (64, T, D)).astype(np.float32), None

    def rep():
        for i in range(0, len(x), max(1, len(x) // 200)):
            yield [x[i:i + 1]]

    keras_out = modelo.predict(x, verbose=0, batch_size=256)
    filas = []
    for variante in ("f32", "dinamica", "int8"):
        fila = {"variante": variante}
        try:
            tfl = convertir(modelo, T, D, variante, rep if variante == "int8" else None)
            out = inferir_tflite(tfl, x)
            fila.update({
                "ok": True, "kb": round(len(tfl) / 1024, 1), "flex": ops_flex(tfl),
                "n_ops": len(ops_del_modelo(tfl)),
                "max_abs_vs_keras": float(np.abs(out - keras_out).max()),
                "top1_igual_keras": float((out.argmax(1) == keras_out.argmax(1)).mean()),
            })
            if y is not None:
                fila["accuracy_datos_vistos"] = float((out.argmax(1) == y).mean())
        except Exception as e:  # se reporta, no se esconde: es justo lo que se mide
            fila.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        filas.append(fila)
        print(json.dumps(fila, ensure_ascii=False))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"tf": tf.__version__, "modelo": args.modelo or "aleatorio", "n": len(x),
                       "resultados": filas}, f, indent=2, ensure_ascii=False)
    fiel = any(f["ok"] and not f["flex"] and f["variante"] == "dinamica" and f["top1_igual_keras"] >= 0.99
               for f in filas)
    print("\nVEREDICTO:", "export sin Flex y fiel a Keras" if fiel else "NO hay export sin Flex fiel")
    sys.exit(0 if fiel else 1)


if __name__ == "__main__":
    main()
