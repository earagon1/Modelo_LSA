"""Entrena el modelo para publicar en la app con todos los datos (contrato v2).

A diferencia de `validar.py`, acá no se reserva ninguna persona: el modelo que va
a la app aprovecha todo lo grabado. Lo que se reporta de él (accuracy sobre datos
vistos) no sirve como evaluación; la evaluación honesta es la de `validar.py`.

Escribe en runs/<fecha>_final/ (no toca models/):
  modelo.keras, actions_15_opt.tflite (dinámica, sin Flex), words.json,
  paridad_modelo.json (fixture para ModeloParidadTest de la app), entrenamiento.json

Después se publica en la app con `scripts/publicar_modelo.py`, que es el único
camino para escribir sus assets (el fixture va a app/src/androidTest/assets/).

Uso, desde la raíz de modelo_LSA:
    python scripts/entrenar_final.py [--semilla 0] [--sin-nada]
"""
import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validar import entrenar

from lsa import dataset as ds
from lsa.export import convertir, inferir_tflite, ops_flex

POR_CLASE_FIXTURE = 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--sin-nada", action="store_true")
    ap.add_argument("--salida", default=os.path.join(ds.RAIZ, "runs", f"{date.today().isoformat()}_final"))
    args = ap.parse_args()

    d = ds.cargar(incluir_nada=not args.sin_nada)
    print(d.resumen(), flush=True)
    modelo = entrenar(d.X, d.y, len(d.clases), args.semilla, args.epocas, verbose=2)

    tfl = convertir(modelo, d.X.shape[1], d.X.shape[2], "dinamica")
    assert not ops_flex(tfl), "quedaron ops Flex"
    p_keras = modelo.predict(d.X, verbose=0, batch_size=256)
    p_tfl = inferir_tflite(tfl, d.X)

    os.makedirs(args.salida, exist_ok=True)
    modelo.save(os.path.join(args.salida, "modelo.keras"))
    with open(os.path.join(args.salida, "actions_15_opt.tflite"), "wb") as f:
        f.write(tfl)
    # words.json con el mismo esquema que el de models/: el orden es el índice de salida.
    with open(os.path.join(args.salida, "words.json"), "w", encoding="utf-8") as f:
        json.dump({"word_ids": d.clases}, f, ensure_ascii=False)

    # Fixture de paridad para el test instrumentado: tomas reales de teléfono de cada
    # clase, con las probabilidades que da Python sobre la entrada ya redondeada.
    idx = [i for c in range(len(d.clases))
           for i in np.where((d.y == c) & (d.fuente == "telefono"))[0][:POR_CLASE_FIXTURE]]
    entradas = np.round(d.X[idx], 7).astype(np.float32)
    probs = inferir_tflite(tfl, entradas)
    fixture = {"modelo_sha256": hashlib.sha256(tfl).hexdigest(), "word_ids": d.clases, "tolerancia": 1e-4,
               "casos": [{"esperado_label": d.clases[int(d.y[i])], "entrada": e.tolist(), "probs": p.tolist()}
                         for i, e, p in zip(idx, entradas, probs, strict=True)]}
    with open(os.path.join(args.salida, "paridad_modelo.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(fixture, f, separators=(",", ":"))

    info = {
        "fecha": date.today().isoformat(), "semilla": args.semilla, "datos": d.resumen(),
        "clases": d.clases, "muestras_por_clase": {d.clases[k]: v for k, v in sorted(Counter(d.y.tolist()).items())},
        "tflite_kb": round(len(tfl) / 1024, 1), "tflite_sha256": hashlib.sha256(tfl).hexdigest(),
        "tflite_top1_igual_keras": float((p_tfl.argmax(1) == p_keras.argmax(1)).mean()),
        "accuracy_datos_vistos": float((p_tfl.argmax(1) == d.y).mean()),
        "aviso": "accuracy sobre datos vistos: no es una evaluación (ver resultados_validacion_v2/)",
    }
    with open(os.path.join(args.salida, "entrenamiento.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\n-> {args.salida}")


if __name__ == "__main__":
    main()
