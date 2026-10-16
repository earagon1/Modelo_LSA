"""Calibración del modelo y elección del umbral de confianza de la app (PY-E4).

El modelo original mostraba el 83 % de sus predicciones con confianza ≥ 0,7 y acertaba
el 32 % de esas: la confianza no significaba nada. Acá se mide y se corrige eso sobre
personas no vistas (el escenario real de la app):

  1. Para cada persona p: se entrena con las demás, reservando un 20 % de sus datos
     para calibrar (nunca de p).
  2. Temperature scaling: una sola temperatura T que reescala las probabilidades,
     ajustada minimizando la log-verosimilitud en el set de calibración.
  3. Sobre p: ECE (error de calibración esperado, 15 bins) y log-loss, con y sin T.
  4. Curva cobertura/acierto: para cada umbral, qué fracción de las señas se muestra
     y cuántas de esas son correctas. Una predicción de "nada" cuenta como no mostrada.

Con eso se elige la T que va a la app (la mediana entre pliegues) y el umbral
(el menor que logra el acierto objetivo entre lo que se muestra).

Escribe en resultados_calibracion/<fecha>_<etiqueta>/. No toca models/ ni la app.

Uso, desde la raíz de modelo_LSA:
    python scripts/calibrar.py --semillas 0 1 2 --acierto-objetivo 0.85
"""
import argparse
import itertools
import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validar import entrenar

from lsa import dataset as ds

UMBRALES = np.round(np.arange(0.30, 0.99, 0.05), 2)


def con_temperatura(p, t):
    """Reescala probabilidades softmax con temperatura t (logits = log p)."""
    z = np.log(np.clip(p, 1e-12, 1.0)) / t
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def ajustar_temperatura(p, y):
    from scipy.optimize import minimize_scalar

    def nll(log_t):
        q = con_temperatura(p, np.exp(log_t))
        return -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, 1.0)))
    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(r.x))


def ece(p, y, bins=15):
    conf = p.max(1)
    ok = p.argmax(1) == y
    total = 0.0
    bordes = np.linspace(0, 1, bins + 1)
    for a, b in itertools.pairwise(bordes):
        m = (conf > a) & (conf <= b)
        if m.any():
            total += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(total)


def log_loss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))


def cobertura(p, y, nada):
    """Por umbral: fracción mostrada y acierto entre lo mostrado ("nada" no se muestra)."""
    conf, pred = p.max(1), p.argmax(1)
    filas = []
    for u in UMBRALES:
        mostrada = (conf >= u) & (pred != nada)
        filas.append({"umbral": float(u), "mostradas": float(mostrada.mean()),
                      "acierto_mostradas": float((pred[mostrada] == y[mostrada]).mean()) if mostrada.any() else None})
    return filas


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--acierto-objetivo", type=float, default=0.85)
    ap.add_argument("--etiqueta", default="lopo")
    args = ap.parse_args()

    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    d = ds.cargar(incluir_nada=True)
    nada = d.clases.index(ds.NADA)
    tel = d.fuente == "telefono"
    personas = sorted(set(d.persona[tel].tolist()))
    print(d.resumen(), flush=True)

    pliegues, todas_p, todas_pt, todas_y = [], [], [], []
    for semilla in args.semillas:
        for p in personas:
            otros = np.where(d.persona != p)[0]
            tr, cal = train_test_split(otros, test_size=0.2, random_state=semilla, stratify=d.y[otros])
            modelo = entrenar(d.X[tr], d.y[tr], len(d.clases), semilla, 300)
            p_cal = modelo.predict(d.X[cal], verbose=0)
            t = ajustar_temperatura(p_cal, d.y[cal])
            te = tel & (d.persona == p)
            y = d.y[te]
            p_raw = modelo.predict(d.X[te], verbose=0)
            p_t = con_temperatura(p_raw, t)
            f = {"semilla": semilla, "persona": p, "temperatura": t,
                 "accuracy": float((p_raw.argmax(1) == y).mean()),
                 "ece_sin": ece(p_raw, y), "ece_con": ece(p_t, y),
                 "logloss_sin": log_loss(p_raw, y), "logloss_con": log_loss(p_t, y)}
            pliegues.append(f)
            todas_p.append(p_raw)
            todas_pt.append(p_t)
            todas_y.append(y)
            print(f"  semilla {semilla} {p}: T={t:.2f} acc {f['accuracy']:.3f} "
                  f"ECE {f['ece_sin']:.3f} -> {f['ece_con']:.3f}", flush=True)

    P, PT, Y = np.concatenate(todas_p), np.concatenate(todas_pt), np.concatenate(todas_y)
    curva_sin, curva_con = cobertura(P, Y, nada), cobertura(PT, Y, nada)
    t_app = float(np.median([f["temperatura"] for f in pliegues]))
    candidatos = [c for c in curva_con if c["acierto_mostradas"] is not None
                  and c["acierto_mostradas"] >= args.acierto_objetivo]
    umbral = candidatos[0] if candidatos else max(curva_con, key=lambda c: c["acierto_mostradas"] or 0)
    con_07 = next(c for c in curva_sin if abs(c["umbral"] - 0.7) < 1e-9)

    r = {"config": vars(args), "datos": d.resumen(), "pliegues": pliegues,
         "temperatura_app": t_app, "umbral_app": umbral, "umbral_actual_sin_calibrar": con_07,
         "ece_sin": ece(P, Y), "ece_con": ece(PT, Y),
         "curva_sin_calibrar": curva_sin, "curva_calibrada": curva_con,
         "confiabilidad": {"sin": confiabilidad(P, Y), "con": confiabilidad(PT, Y)}}
    salida = os.path.join(ds.RAIZ, "resultados_calibracion", f"{date.today().isoformat()}_{args.etiqueta}")
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)
    with open(os.path.join(ds.RAIZ, "resultados_calibracion", "calibracion_app.json"), "w", encoding="utf-8") as f:
        json.dump({"temperatura": round(t_app, 4), "umbral": umbral["umbral"],
                   "origen": os.path.relpath(salida, ds.RAIZ).replace(os.sep, "/")}, f, indent=2)

    pct = lambda v: "—" if v is None else f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
    md = ["# Calibración sobre personas no vistas", "", f"Datos: {r['datos']}. Semillas: {args.semillas}.", "",
          f"- Temperatura para la app (mediana de {len(pliegues)} pliegues): **T = {t_app:.2f}**",
          f"- ECE: {pct(r['ece_sin'])} % sin calibrar → {pct(r['ece_con'])} % calibrado",
          f"- Con el umbral actual de la app (0,70, sin calibrar): se muestra el {pct(con_07['mostradas'])} % "
          f"y acierta el {pct(con_07['acierto_mostradas'])} % de lo mostrado.",
          f"- Umbral elegido (acierto objetivo {pct(args.acierto_objetivo)} %): **{umbral['umbral']:.2f}** sobre la "
          f"confianza calibrada: muestra el {pct(umbral['mostradas'])} % y acierta el "
          f"{pct(umbral['acierto_mostradas'])} % de lo mostrado.", "",
          "| Umbral | Mostradas sin calibrar | Acierto | Mostradas calibrado | Acierto |", "|---|---|---|---|---|"]
    for a, b in zip(curva_sin, curva_con, strict=True):
        md.append(f"| {a['umbral']:.2f} | {pct(a['mostradas'])} | {pct(a['acierto_mostradas'])} "
                  f"| {pct(b['mostradas'])} | {pct(b['acierto_mostradas'])} |")
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {salida}")


def confiabilidad(p, y, bins=10):
    """Datos del diagrama de confiabilidad: confianza media contra acierto por bin."""
    conf, ok = p.max(1), p.argmax(1) == y
    out = []
    for a, b in itertools.pairwise(np.linspace(0, 1, bins + 1)):
        m = (conf > a) & (conf <= b)
        if m.any():
            out.append({"desde": float(a), "hasta": float(b), "n": int(m.sum()),
                        "confianza": float(conf[m].mean()), "acierto": float(ok[m].mean())})
    return out


if __name__ == "__main__":
    main()
