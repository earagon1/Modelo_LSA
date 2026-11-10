"""Cuánto acierto cuesta la privacidad diferencial central (R6 de plan/federado_robustez.md).

DP-FedAvg como lo hace `servidor.py --dp-ruido z`: cada update recortado a norma C,
promedio sin ponderar, ruido N(0, (z·C/k)²) por coordenada. El ε sale de
`agregacion.epsilon_zcdp` (persona entera, todos en cada ronda, δ = 1e-5).

El ruido se divide por k, la cantidad de clientes, y la señal no: con pocos clientes, el
ruido que da un ε razonable tapa el update. Para ver cómo cambia con k, cada persona del
piloto se parte en S sub-clientes (S = 1, 5, 10 → 3, 15 y 30 clientes; mismas tomas,
repartidas en más teléfonos).

C = 0,7 está fijo de antemano: es la norma honesta mediana que midió
medir_limite_y_rotacion.py el 2026-10-18 (0,68-0,76). Elegirlo mirando los updates de
esta corrida gastaría privacidad que no se cuenta.

Además de la grilla medida, el resumen da una estimación (no una medición) de cuántos
clientes harían falta para que el ruido sea el 10 % de la señal: la norma del ruido es
z·C·√d / k, con d los parámetros de la cabeza.

Uso, desde la raíz de modelo_LSA:
    python federated/medir_privacidad.py --dir federated/piloto
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import agregacion as ag  # noqa: E402
import protocolo as pr  # noqa: E402
from medir_agregacion_robusta import Entrenador, cargar_piloto, evaluar  # noqa: E402

CLIP = 0.7
RUIDOS = (0.0, 0.1, 0.3, 1.0, 2.0, 4.0)


def correr_dp(entrenador, pesos0, clientes, ruido, rondas, epocas, semilla, evaluacion, clip=CLIP):
    """`ruido` = None: FedAvg ponderado de siempre (la referencia sin DP ni recorte)."""
    pesos = [np.asarray(w, np.float32) for w in pesos0]
    rng = np.random.default_rng(semilla + 104729)
    curva, senal, recortados = [], [], []
    for r in range(rondas):
        updates = [(entrenador.delta(pesos, e, y, epocas, semilla * 1000 + 31 * r + i), len(y))
                   for i, (e, y) in enumerate(clientes)]
        if ruido is None:
            pesos = pr.agregar(pesos, updates)
        else:
            sin_ruido, _ = ag.promedio_privado(updates, clip, 0.0, rng)
            combinado, det = ag.promedio_privado(updates, clip, ruido, rng)
            senal.append(pr.norma(sin_ruido))
            recortados.append(det["recortados"] / det["clientes"])
            pesos = [(w + c).astype(np.float32) for w, c in zip(pesos, combinado, strict=True)]
        curva.append(evaluar(pesos, *evaluacion)["acierto"])
    final = evaluar(pesos, *evaluacion)
    final.update({"curva": curva, "clientes": len(clientes),
                  "epsilon": None if ruido is None else ag.epsilon_zcdp(rondas, ruido),
                  "norma_senal_media": float(np.mean(senal)) if senal else None,
                  "fraccion_recortada": float(np.mean(recortados)) if recortados else None})
    return final


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--sub-clientes", nargs="+", type=int, default=[1, 5, 10])
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rondas", type=int, default=5)
    ap.add_argument("--epocas", type=int, default=2)
    args = ap.parse_args()
    # Con la salida redirigida a un archivo, Windows usa cp1252 y "ε" o "√" tiran la corrida.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    n, pesos0, clases, clientes_con, evaluacion = cargar_piloto(args.dir)
    d = int(sum(w.size for w in pesos0))
    entrenador = Entrenador(128, len(clases))
    salida = os.path.join(RAIZ, "resultados_federado_privacidad", date.today().isoformat())
    os.makedirs(salida, exist_ok=True)
    filas, t0 = [], time.time()
    for s in args.sub_clientes:
        clientes = clientes_con(s)
        for semilla in args.semillas:
            for ruido in (None, *RUIDOS):
                r = correr_dp(entrenador, pesos0, clientes, ruido, args.rondas, args.epocas, semilla, evaluacion)
                filas.append({"sub_clientes": s, "semilla": semilla, "ruido": ruido, **r})
                eps = "-" if r["epsilon"] is None else f"{r['epsilon']:.1f}"
                print(f"[{time.time() - t0:5.0f} s] k={len(clientes):2d} s{semilla} z={ruido} ε={eps:>7s} "
                      f"{100 * r['acierto']:5.1f} %", flush=True)
        with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "paquete": n, "clip": CLIP, "delta": ag.DELTA_DP,
                       "parametros_cabeza": d, "filas": filas}, fh, indent=1, ensure_ascii=False)
    escribir_resumen(salida, filas, args, d)
    print(f"\n-> {salida} ({(time.time() - t0) / 60:.1f} min)")


def escribir_resumen(salida, filas, args, d):
    pct = lambda v: f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
    num = lambda v: f"{v:.1f}".replace(".", ",")  # noqa: E731
    ks = sorted({f["clientes"] for f in filas})
    md = ["# Privacidad diferencial central: acierto contra ε", "",
          f"DP-FedAvg, {args.rondas} rondas, C = {num(CLIP)}, δ = 1e-5, semillas {args.semillas}. Acierto sobre la "
          "sesión 2 (media ± desvío entre semillas). La fila «sin DP» es FedAvg ponderado, sin recorte ni ruido.",
          "", "| z | ε | " + " | ".join(f"{k} clientes" for k in ks) + " |", "|---|---|" + "---|" * len(ks)]
    for ruido in (None, *RUIDOS):
        celdas = []
        for k in ks:
            v = [f["acierto"] for f in filas if f["clientes"] == k and f["ruido"] == ruido]
            celdas.append(f"{pct(np.mean(v))} ± {pct(np.std(v))}" if v else "")
        eps = "-" if ruido is None else ("∞" if ruido == 0 else num(ag.epsilon_zcdp(args.rondas, ruido)))
        md.append(f"| {'sin DP' if ruido is None else num(ruido)} | {eps} | " + " | ".join(celdas) + " |")
    senal = float(np.mean([f["norma_senal_media"] for f in filas if f["ruido"] == 0.0]))
    md += ["", f"Norma media del promedio recortado sin ruido (la señal): {senal:.3f}. Parámetros de la cabeza: "
           f"d = {d}.", "",
           "**Estimación, no medición:** clientes necesarios para que la norma del ruido (z·C·√d / k) sea el 10 % "
           "de esa señal:", "", "| z | ε (5 rondas) | clientes |", "|---|---|---|"]
    for ruido in RUIDOS[1:]:
        k_nec = ruido * CLIP * np.sqrt(d) / (0.1 * senal)
        md.append(f"| {num(ruido)} | {num(ag.epsilon_zcdp(args.rondas, ruido))} | {int(np.ceil(k_nec))} |")
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
