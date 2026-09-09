"""
Compara las variantes de cuantizacion del modelo .tflite.

La propuesta compromete evaluar dos tecnicas de cuantizacion post-entrenamiento
(dinamica y entera) pero nunca se midieron: `export_tflite.py` generaba la
dinamica y tenia la entera desactivada, y la app terminaba cargando igual el
float32. Este script produce la tabla que faltaba.

Para cada variante mide:

  - Tamano en disco, que es lo que ocupa en el APK.
  - Accuracy y F1-macro sobre el mismo conjunto de test que usa el experimento
    federado (split estratificado, seed 42), asi los numeros son comparables.
  - Latencia por inferencia con batch 1 y 4 hilos, que es como corre en el
    telefono. Se reporta la mediana y el percentil 95: el p95 importa mas que
    el promedio porque un pico ocasional se siente como un tiron en la captura.
  - Cuantas predicciones cambian respecto del float32. Es la medida directa de
    lo que cuesta cuantizar: dos modelos pueden tener la misma accuracy y aun
    asi discrepar en que muestras aciertan.

AVISO SOBRE LOS NUMEROS ABSOLUTOS
---------------------------------
`actions_15.keras` se entreno con todo el dataset, asi que el "test" de aca no
es material no visto y la accuracy sale optimista. La comparacion ENTRE
variantes sigue siendo valida, que es para lo que existe este script: todas
parten del mismo modelo y ven exactamente los mismos datos. No copiar la
accuracy absoluta al informe como si fuera generalizacion.

Uso:
    python modelo_LSA/benchmark_cuantizacion.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # latencia CPU, como el telefono

import numpy as np

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "federated"))

import tensorflow as tf  # noqa: E402

from data import cargar_dataset, split_train_test  # noqa: E402
from metrics import classification_metrics  # noqa: E402

MODELOS = [
    ("float32", "actions_15.tflite"),
    ("dinamica (int8 pesos)", "actions_15_opt.tflite"),
    ("entera (int8 + calibracion)", "actions_15_int8.tflite"),
]

REPETICIONES_LATENCIA = 3
CALENTAMIENTO = 10


def predecir(ruta_modelo: Path, X: np.ndarray, num_threads: int = 4):
    """
    Corre el .tflite muestra por muestra y devuelve (probs, latencias_ms).

    Batch 1 a proposito: en la app cada sena se infiere sola, no en lote, asi
    que un promedio sobre batches grandes no representaria nada.
    """
    interprete = tf.lite.Interpreter(model_path=str(ruta_modelo), num_threads=num_threads)
    interprete.allocate_tensors()

    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]

    def una(x):
        interprete.set_tensor(entrada["index"], x[None, ...].astype(entrada["dtype"]))
        interprete.invoke()
        return interprete.get_tensor(salida["index"])[0]

    # Calentamiento: las primeras invocaciones pagan la construccion de los
    # kernels y ensucian la mediana.
    for i in range(min(CALENTAMIENTO, len(X))):
        una(X[i])

    probs = np.stack([una(x) for x in X], axis=0)

    latencias = []
    for _ in range(REPETICIONES_LATENCIA):
        for x in X:
            t0 = time.perf_counter()
            una(x)
            latencias.append((time.perf_counter() - t0) * 1000.0)

    return probs, np.asarray(latencias)


def main():
    modelos_dir = AQUI / "models"

    # Indices de words.json completo: el .keras se entreno con 13 salidas y
    # renumerar las clases presentes correria las etiquetas posteriores al hueco.
    X, y, etiquetas = cargar_dataset(conservar_indices_de_words_json=True, verbose=False)
    _, X_test, _, y_test = split_train_test(X, y, test_size=0.2, seed=42)
    print(f"Test: {len(X_test)} muestras, {len(etiquetas)} clases\n")

    filas = []
    probs_f32 = None

    for nombre, archivo in MODELOS:
        ruta = modelos_dir / archivo
        if not ruta.exists():
            print(f"[falta] {archivo} -- corre export_tflite.py primero")
            continue

        print(f"midiendo {nombre} ({archivo})...")
        probs, latencias = predecir(ruta, X_test)
        m = classification_metrics(y_test, probs)

        if probs_f32 is None:
            probs_f32 = probs
            discrepancia = 0.0
        else:
            distintas = (probs.argmax(1) != probs_f32.argmax(1)).sum()
            discrepancia = 100.0 * distintas / len(probs)

        filas.append({
            "nombre": nombre,
            "kb": ruta.stat().st_size / 1024,
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "mediana": float(np.median(latencias)),
            "p95": float(np.percentile(latencias, 95)),
            "discrepancia": discrepancia,
        })

    if not filas:
        print("No se pudo medir ningun modelo.")
        return

    base_kb = filas[0]["kb"]
    base_acc = filas[0]["accuracy"]

    cab = (
        "| Variante | Tamano | vs f32 | Accuracy | F1-macro | Latencia mediana | p95 | Predicciones distintas |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    cuerpo = []
    for f in filas:
        cuerpo.append(
            f"| {f['nombre']} | {f['kb']:.0f} KB | {f['kb'] / base_kb:.2f}x | "
            f"{f['accuracy']:.4f} | {f['f1_macro']:.4f} | "
            f"{f['mediana']:.2f} ms | {f['p95']:.2f} ms | "
            f"{f['discrepancia']:.1f} % |"
        )
    tabla = cab + "\n".join(cuerpo)

    print("\n" + tabla)

    print("\nLECTURA")
    print("=" * 72)
    for f in filas[1:]:
        d_acc = (f["accuracy"] - base_acc) * 100
        ahorro = (1 - f["kb"] / base_kb) * 100
        veloc = filas[0]["mediana"] / f["mediana"] if f["mediana"] > 0 else float("nan")
        print(
            f"{f['nombre']}: ocupa {ahorro:.0f} % menos, corre {veloc:.2f}x, "
            f"accuracy {d_acc:+.2f} puntos, cambia {f['discrepancia']:.1f} % de las predicciones"
        )
    print(
        "\nRecordar: la accuracy absoluta es optimista (el modelo vio estos datos\n"
        "al entrenar). Lo comparable es la diferencia entre variantes."
    )

    destino = modelos_dir / "benchmark_cuantizacion.md"
    destino.write_text(tabla + "\n", encoding="utf-8")
    print(f"\nTabla guardada en {destino}")


if __name__ == "__main__":
    main()
