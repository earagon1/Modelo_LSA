"""Validación honesta del modelo con todas las fuentes (PY-E3).

Esquemas (cada uno contesta otra pregunta):
  publicado          ¿Cuánto acierta HOY la app con personas nuevas? El .tflite de la app
                     sobre la secuencia exacta que la app calculó en cada toma de la sesión.
  webcam_a_telefono  ¿Alcanza con la webcam? Entrena solo con la webcam (como el modelo
                     publicado, pero con el contrato v2) y evalúa en los teléfonos.
  persona            ¿Generaliza a alguien que no vio? Leave-one-person-out: entrena con
                     todos menos una persona (sus tomas de webcam incluidas) y evalúa con ella.
  sesion             ¿Generaliza a otra sesión? StratifiedGroupKFold agrupando por sesión.
  estratificado      El esquema viejo (tomas mezcladas al azar). Solo para cuantificar
                     cuánto inflaba los números la similitud entre tomas.

Escribe en resultados_validacion_v2/<fecha>_<etiqueta>/ y no toca models/.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH, ver CLAUDE.md):
    python scripts/validar.py --etiqueta p1p2
    python scripts/validar.py --esquemas publicado persona --semillas 0 1 2
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa import dataset as ds

RAIZ = ds.RAIZ
ESQUEMAS = ["publicado", "webcam_a_telefono", "persona", "sesion", "estratificado"]
APP_TFLITE = os.path.normpath(os.path.join(RAIZ, "..", "..", "TraductorLSA", "app", "src", "main", "assets",
                                           "actions_15_opt.tflite"))


def metricas(y_true, y_pred, clases):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    presentes = sorted(set(y_true.tolist()))
    f1_clase = f1_score(y_true, y_pred, labels=presentes, average=None, zero_division=0)
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=presentes, average="macro", zero_division=0)),
        "f1_por_clase": {clases[c]: round(float(f), 4) for c, f in zip(presentes, f1_clase, strict=True)},
        "matriz": confusion_matrix(y_true, y_pred, labels=list(range(len(clases)))).tolist(),
    }


def entrenar(X, y, n_clases, semilla, epocas, verbose=0):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight

    from model import get_model

    tf.keras.utils.set_random_seed(semilla)
    # La validación para el early stopping sale del propio train: el test no se toca.
    estrat = y if min(np.bincount(y)[np.bincount(y) > 0]) >= 2 else None
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.15, random_state=semilla, stratify=estrat)
    presentes = np.unique(ytr)
    pesos = dict(zip(presentes.tolist(), compute_class_weight("balanced", classes=presentes, y=ytr), strict=True))
    modelo = get_model(X.shape[1], n_clases)
    a_onehot = lambda v: tf.keras.utils.to_categorical(v, n_clases)  # noqa: E731
    modelo.fit(Xtr, a_onehot(ytr), validation_data=(Xva, a_onehot(yva)), epochs=epocas, batch_size=32,
               class_weight=pesos, verbose=verbose,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25,
                                                           restore_best_weights=True)])
    return modelo


def acumular(destino, d, mascara_test, pred):
    for i, p in zip(np.where(mascara_test)[0], pred, strict=True):
        destino.append({"y": int(d.y[i]), "pred": int(p), "persona": str(d.persona[i]),
                        "fuente": str(d.fuente[i]), "dispositivo": str(d.dispositivo[i])})


def resumir(predicciones, clases):
    y = np.array([p["y"] for p in predicciones])
    pr = np.array([p["pred"] for p in predicciones])
    out = {"global": metricas(y, pr, clases), "por_persona": {}, "por_fuente": {}}
    for clave, campo in (("por_persona", "persona"), ("por_fuente", "fuente")):
        for valor in sorted({p[campo] for p in predicciones}):
            m = np.array([p[campo] == valor for p in predicciones])
            r = metricas(y[m], pr[m], clases)
            out[clave][valor] = {k: r[k] for k in ("n", "accuracy", "f1_macro")}
    return out


def esquema_publicado(clases_modelo, umbral=0.7):
    """El .tflite de la app sobre la secuencia v1 que la app guardó en cada toma de la sesión."""
    from lsa.export import inferir_tflite
    with open(APP_TFLITE, "rb") as f:
        tfl = f.read()
    indice = {c: i for i, c in enumerate(clases_modelo)}
    predicciones, confianzas = [], []
    for ruta in sorted(glob.glob(os.path.join(RAIZ, "data", "captura_E3", "*", "lsa_samples.json"))):
        persona = os.path.basename(os.path.dirname(ruta))
        v2 = os.path.join(os.path.dirname(ruta), "lsa_samples_v2.jsonl")
        fechas = {date.fromtimestamp(m["ts"] / 1000).isoformat() for m in ds.dataset_v2.leer(v2) if m.get("sesion")}
        with open(ruta, encoding="utf-8") as f:
            v1 = json.load(f)
        muestras = [m for fecha, lista in v1["by_date"].items() if fecha in fechas for m in lista
                    if m["label"] in indice]
        if not muestras:
            continue
        X = np.asarray([m["seq"] for m in muestras], np.float32)
        p = inferir_tflite(tfl, X)
        for m, probs in zip(muestras, p, strict=True):
            predicciones.append({"y": indice[m["label"]], "pred": int(probs.argmax()), "persona": persona,
                                 "fuente": "telefono", "dispositivo": persona})
            confianzas.append(float(probs.max()))
    r = resumir(predicciones, clases_modelo)
    conf = np.array(confianzas)
    ok = np.array([p["y"] == p["pred"] for p in predicciones])
    r["umbral_app"] = umbral
    r["mostradas"] = float((conf >= umbral).mean())
    r["accuracy_de_las_mostradas"] = float(ok[conf >= umbral].mean()) if (conf >= umbral).any() else None
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esquemas", nargs="+", default=ESQUEMAS, choices=ESQUEMAS)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0])
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--sin-nada", action="store_true", help="no entrenar la clase 'nada'")
    ap.add_argument("--etiqueta", default="run")
    args = ap.parse_args()

    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    salida = os.path.join(RAIZ, "resultados_validacion_v2", f"{date.today().isoformat()}_{args.etiqueta}")
    os.makedirs(salida, exist_ok=True)
    d = ds.cargar(incluir_nada=not args.sin_nada)
    print(d.resumen(), flush=True)
    resultados = {"datos": d.resumen(), "clases": d.clases, "semillas": args.semillas, "esquemas": {}}
    C = len(d.clases)

    for esquema in args.esquemas:
        t0 = time.time()
        if esquema == "publicado":
            resultados["esquemas"][esquema] = esquema_publicado(ds.clases_del_modelo(incluir_nada=False))
        else:
            predicciones = []
            for semilla in args.semillas:
                if esquema == "webcam_a_telefono":
                    tr = d.fuente == "webcam"
                    te = (d.fuente == "telefono") & (np.array(d.clases)[d.y] != ds.NADA)
                    pliegues = [(tr, te)]
                elif esquema == "persona":
                    pliegues = [(d.persona != p, (d.persona == p) & (d.fuente == "telefono"))
                                for p in sorted(set(d.persona[d.fuente == "telefono"].tolist()))]
                elif esquema == "sesion":
                    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=semilla)
                    pliegues = [(np.isin(np.arange(len(d.y)), a), np.isin(np.arange(len(d.y)), b))
                                for a, b in kf.split(d.X, d.y, d.sesion)]
                else:
                    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
                    pliegues = [(np.isin(np.arange(len(d.y)), a), np.isin(np.arange(len(d.y)), b))
                                for a, b in kf.split(d.X, d.y)]
                for k, (tr, te) in enumerate(pliegues):
                    modelo = entrenar(d.X[tr], d.y[tr], C, semilla, args.epocas)
                    pred = modelo.predict(d.X[te], verbose=0).argmax(1)
                    acumular(predicciones, d, te, pred)
                    print(f"  {esquema} semilla {semilla} pliegue {k + 1}/{len(pliegues)}: "
                          f"acc {np.mean(pred == d.y[te]):.3f} (n={te.sum()})", flush=True)
            resultados["esquemas"][esquema] = resumir(predicciones, d.clases)
        g = resultados["esquemas"][esquema]["global"]
        print(f"{esquema}: accuracy {g['accuracy']:.3f}, F1 macro {g['f1_macro']:.3f} "
              f"({time.time() - t0:.0f} s)", flush=True)

    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    md = [f"# Validación v2 ({resultados['datos']})", "", "| Esquema | n | Accuracy | F1 macro | Por persona |",
          "|---|---|---|---|---|"]
    for esquema, r in resultados["esquemas"].items():
        g = r["global"]
        pp = ", ".join(f"{p}: {v['accuracy']:.2f}" for p, v in r["por_persona"].items())
        md.append(f"| {esquema} | {g['n']} | {g['accuracy']:.3f} | {g['f1_macro']:.3f} | {pp} |")
    if "publicado" in resultados["esquemas"]:
        r = resultados["esquemas"]["publicado"]
        md += ["", f"Modelo publicado: la app muestra el {100 * r['mostradas']:.0f} % de las predicciones "
               f"(confianza ≥ {r['umbral_app']}); de esas, acierta el "
               f"{100 * (r['accuracy_de_las_mostradas'] or 0):.0f} %."]
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
