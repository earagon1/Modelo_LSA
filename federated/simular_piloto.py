"""Repite en la computadora las rondas del piloto federado y las compara con las reales.

El piloto sin P2 (`federated/piloto_sinP2`) corrió 4 rondas con dos dispositivos: el
OnePlus de P2 (84 tomas) y la tableta de P1 (93). Este script hace lo mismo sin teléfonos:
parte de la cabeza del paquete, entrena a cada cliente con sus tomas como lo hace la app
(`cliente_emulado.entrenar_cabeza`, el espejo de CabezaDensa.kt), sube el delta a 8 bits y
lo agrega con FedAvg (`protocolo.agregar`), igual que el servidor.

Los clientes salen de las mismas reglas que la app (`ClienteFederado.esAportable`): tomas
de teléfono de esa persona, fuera del test fijo del servidor y fuera de la sesión de
robustez. Si la cantidad no coincide con la que registró el piloto, el script se detiene:
no estaría simulando lo mismo.

Mide lo mismo que el piloto, para P2: sus tomas del test fijo (las habituales, que nunca
entrenan) y su sesión 2 (bloque B). También evalúa las cabezas reales de cada ronda, para
tener las dos cosas lado a lado, y repite la simulación con varias semillas (el
entrenamiento local y la cuantización son aleatorios) y con más épocas locales.

Uso, desde la raíz de modelo_LSA:
    python federated/simular_piloto.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import protocolo as pr  # noqa: E402

PILOTO = os.path.join(AQUI, "piloto_sinP2")
# Lo que registró el piloto (rondas.jsonl): quién entrenó y con cuántas tomas.
CLIENTES = {"P2": ("p1_8b78a18690edf9c7", "OnePlus CPH2663"), "P1": ("p1_b80dde3cced831aa", "LENOVO TB321FU")}
EVALUADA = "P2"


def rondas_reales(directorio):
    with open(os.path.join(directorio, "rondas.jsonl"), encoding="utf-8") as f:
        filas = [json.loads(linea) for linea in f if linea.strip()]
    return [r for r in filas if r.get("evento") == "ronda"]


def tomas_de_cliente(d, persona, test_fijo):
    """Las tomas que la app usa para entrenar: de teléfono, de esa persona, fuera del test fijo.
    La sesión de robustez (bloque B) ya no está porque se carga solo el bloque de entrenamiento."""
    return (d.fuente == "telefono") & (d.persona == persona) & ~np.isin(d.uuid, list(test_fijo))


def acierto(pesos, emb, y):
    return float(np.mean(pr.predecir(pesos, emb).argmax(1) == y)) if len(y) else float("nan")


def una_simulacion(pesos0, clientes, evaluar, rondas, epocas, semilla, entrenar_cabeza):
    """FedAvg de la cabeza como el servidor: cada ronda, todos los clientes, delta a 8 bits."""
    pesos = [w.copy() for w in pesos0]
    curva = [{k: acierto(pesos, e, y) for k, (e, y) in evaluar.items()}]
    n_clases = pesos[-1].shape[0]
    for r in range(rondas):
        updates = []
        for i, (emb, y) in enumerate(clientes):
            locales = entrenar_cabeza(pesos, emb, y, n_clases, epocas, semilla=semilla * 1000 + r * 10 + i)
            delta = [lo - g for lo, g in zip(locales, pesos, strict=True)]
            rng = np.random.default_rng(semilla * 1000 + r * 10 + i)
            delta = pr.decodificar_delta(pr.codificar_delta(delta, rng))
            updates.append((delta, len(y)))
        pesos = pr.agregar(pesos, updates)
        curva.append({k: acierto(pesos, e, y) for k, (e, y) in evaluar.items()})
    return curva


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=PILOTO)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epocas", nargs="+", type=int, default=[2, 8])
    ap.add_argument("--salida", default=os.path.join(RAIZ, "resultados_federado_simulacion_piloto",
                                                     date.today().isoformat()))
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from cliente_emulado import entrenar_cabeza
    from lsa import dataset as ds
    from lsa.export import inferir_tflite

    carpeta = os.path.join(args.dir, "paquetes", "v1")
    with open(os.path.join(carpeta, "backbone.tflite"), "rb") as f:
        tfl = f.read()
    with open(os.path.join(carpeta, "manifiesto.json"), encoding="utf-8") as f:
        clases = json.load(f)["clases"]
    with open(os.path.join(carpeta, "cabeza.json"), encoding="utf-8") as f:
        pesos0, _ = pr.cabeza_desde_json(json.load(f))
    with open(os.path.join(args.dir, "test_fijo.json"), encoding="utf-8") as f:
        test_fijo = set(json.load(f)["uuids"])

    reales = rondas_reales(args.dir)
    rondas = len(reales)
    n_registrado = {p: reales[0]["n_muestras"][pid] for p, (pid, _) in CLIENTES.items()}

    a = ds.cargar(incluir_nada=True, clases=clases)
    b = ds.cargar(incluir_nada=True, bloques=("B",), clases=clases)
    emb_a = inferir_tflite(tfl, a.X).astype(np.float32)
    emb_b = inferir_tflite(tfl, b.X).astype(np.float32)

    clientes, n_simulado = [], {}
    for p in CLIENTES:
        m = tomas_de_cliente(a, p, test_fijo)
        n_simulado[p] = int(m.sum())
        clientes.append((emb_a[m], a.y[m]))
    if n_simulado != n_registrado:
        raise SystemExit(f"Las tomas de los clientes no coinciden con el piloto: {n_simulado} contra {n_registrado}")

    habituales = (a.fuente == "telefono") & (a.persona == EVALUADA) & np.isin(a.uuid, list(test_fijo))
    sesion2 = b.persona == EVALUADA
    evaluar = {"habituales": (emb_a[habituales], a.y[habituales]), "sesion2": (emb_b[sesion2], b.y[sesion2])}
    n_eval = {k: int(len(y)) for k, (_, y) in evaluar.items()}

    # Las cabezas que publicó el piloto real, v1 (paquete) a v5 (tras la cuarta ronda).
    real = []
    for ruta in sorted(glob.glob(os.path.join(args.dir, "cabezas", "v*.json")),
                       key=lambda r: int(re.search(r"v(\d+)", os.path.basename(r)).group(1))):
        with open(ruta, encoding="utf-8") as f:
            pesos, _ = pr.cabeza_desde_json({"clases": clases, "capas": json.load(f)["capas"]})
        v = int(re.search(r"v(\d+)", os.path.basename(ruta)).group(1))
        if v >= 1:
            real.append({"version": v, **{k: acierto(pesos, e, y) for k, (e, y) in evaluar.items()}})

    simulado = {}
    for ep in args.epocas:
        curvas = [una_simulacion(pesos0, clientes, evaluar, rondas, ep, s, entrenar_cabeza) for s in args.semillas]
        simulado[ep] = curvas
        finales = {k: [c[-1][k] for c in curvas] for k in evaluar}
        print(f"{ep} épocas: " + " | ".join(
            f"{k} {100 * np.mean(v):.1f} ± {100 * np.std(v, ddof=1):.1f} (de {100 * min(v):.1f} a {100 * max(v):.1f})"
            for k, v in finales.items()))
    print("real: " + " | ".join(f"v{r['version']} {100 * r['habituales']:.1f}/{100 * r['sesion2']:.1f}" for r in real))

    os.makedirs(args.salida, exist_ok=True)
    resultado = {"piloto": os.path.relpath(args.dir, RAIZ).replace(os.sep, "/"), "rondas": rondas,
                 "tomas_por_cliente": n_simulado, "tomas_evaluadas": n_eval, "semillas": args.semillas,
                 "real": real, "simulado": {str(ep): c for ep, c in simulado.items()}}
    with open(os.path.join(args.salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=1, ensure_ascii=False)
    escribir_resumen(args.salida, resultado)
    print(f"-> {args.salida}")


def escribir_resumen(salida, r):
    pct = lambda v: f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
    he, se = r["tomas_evaluadas"]["habituales"], r["tomas_evaluadas"]["sesion2"]
    md = ["# Simulación de las rondas del piloto federado", "",
          f"Generado por `federated/simular_piloto.py` desde `{r['piloto']}`. No editar a mano.", "",
          f"{r['rondas']} rondas, dos clientes con las mismas tomas que el piloto ("
          + ", ".join(f"{p}: {n}" for p, n in r["tomas_por_cliente"].items()) + "), "
          f"delta a 8 bits y FedAvg. Semillas {r['semillas']}. Se evalúa a P2: sus {he} tomas habituales (del test "
          f"fijo, nunca entrenan) y sus {se} tomas de la sesión 2. Una toma vale {pct(1 / he)} y "
          f"{pct(1 / se)} puntos, respectivamente.", "",
          "## Piloto real, versión por versión", "", "| Versión | Habituales (%) | Sesión 2 (%) |", "|---|---|---|"]
    md += [f"| v{x['version']} | {pct(x['habituales'])} | {pct(x['sesion2'])} |" for x in r["real"]]
    md += ["", "## Simulación, tras la última ronda", "",
           "| Épocas locales | Habituales (%) | Sesión 2 (%) |", "|---|---|---|"]
    for ep, curvas in r["simulado"].items():
        celdas = []
        for k in ("habituales", "sesion2"):
            v = [c[-1][k] for c in curvas]
            celdas.append(f"{pct(np.mean(v))} ± {pct(np.std(v, ddof=1))} (de {pct(min(v))} a {pct(max(v))})")
        md.append(f"| {ep} | {celdas[0]} | {celdas[1]} |")
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
