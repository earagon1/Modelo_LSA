"""¿Mejoró el reconocimiento con las rondas del piloto? Cada versión de la cabeza, sobre
tomas que no participaron ni del entrenamiento ni de las rondas.

Las apps no usan en el federado las tomas de la sesión 2 (bloque B, robustez) ni las
del test fijo del servidor (`OperacionesFederadas.esAportable`). Por eso sirven de
prueba honesta para comparar la cabeza del paquete (v1) con las que salieron de cada
ronda (v2, v3…), con el mismo backbone.

Uso, desde la raíz de modelo_LSA:
    python federated/evaluar_piloto.py --dir federated/piloto
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import protocolo as pr  # noqa: E402
from servidor import ultimo_paquete  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from lsa import dataset as ds
    from lsa.export import inferir_tflite

    n = ultimo_paquete(args.dir)
    carpeta = os.path.join(args.dir, "paquetes", f"v{n}")
    with open(os.path.join(carpeta, "backbone.tflite"), "rb") as f:
        tfl = f.read()
    with open(os.path.join(carpeta, "manifiesto.json"), encoding="utf-8") as f:
        man = json.load(f)
    clases = man["clases"]
    b = ds.cargar(incluir_nada=True, bloques=("B",), clases=clases)
    emb = inferir_tflite(tfl, b.X).astype(np.float32)
    z = np.load(os.path.join(carpeta, "test_servidor.npz"))

    versiones = []
    for ruta in glob.glob(os.path.join(args.dir, "cabezas", "v*.json")):
        with open(ruta, encoding="utf-8") as f:
            d = json.load(f)
        if d.get("paquete", n) != n:
            continue
        pesos, _ = pr.cabeza_desde_json({"clases": clases, "capas": d["capas"]})
        versiones.append((int(re.search(r"v(\d+)", os.path.basename(ruta)).group(1)), pesos))
    versiones.sort()

    personas = sorted(set(b.persona.tolist()))
    print(f"Sesión 2 (nunca usada en el federado): {len(b.y)} tomas | test fijo del servidor: {len(z['etiquetas'])}")
    print("versión | sesión 2 total | " + " | ".join(personas) + " | test fijo")
    filas = []
    for v, pesos in versiones:
        pred = pr.predecir(pesos, emb).argmax(1)
        ok = pred == b.y
        por = {p: float(ok[b.persona == p].mean()) for p in personas}
        fijo = float(np.mean(pr.predecir(pesos, z["embeddings"]).argmax(1) == z["etiquetas"]))
        filas.append({"version": v, "sesion2": float(ok.mean()), "por_persona": por, "test_fijo": fijo})
        print(f"v{v} | {100 * ok.mean():.1f} % | " + " | ".join(f"{100 * por[p]:.1f}" for p in personas)
              + f" | {100 * fijo:.1f} %")
    with open(os.path.join(args.dir, "evaluacion_versiones.json"), "w", encoding="utf-8") as f:
        json.dump({"paquete": n, "n_sesion2": len(b.y), "versiones": filas}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
