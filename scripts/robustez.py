"""Análisis de robustez: cuánto cae el reconocimiento en cada condición adversa (PY-E5).

La sesión 2 grabó las 12 señas una vez en cada condición (poca luz, girado 30°, cerca y
lejos) con las mismas tres personas y los mismos dispositivos de la sesión 1. La
pregunta de la propuesta (§3.5) es cuánto se degrada el sistema fuera de la condición
estándar. Para que la comparación sea justa, cada condición se mide **con el mismo
modelo** que la referencia estándar:

  persona_nueva      Entrena sin la persona evaluada (LOPO, como validar.py) y evalúa
                     con sus tomas de la sesión 1 (estándar) y de la sesión 2 (condiciones).
                     Es el caso de alguien que usa la app por primera vez.
  persona_conocida   Entrena con todo menos la tanda 2 de la persona evaluada y evalúa
                     con esa tanda (estándar, otra sesión) y con la sesión 2. Es el caso
                     de alguien cuyos datos ya están en el modelo (como el publicado).
  app                El modelo publicado en la app, con su temperatura y su umbral,
                     sobre la secuencia que la app calculó en vivo en cada toma de la
                     sesión 2 (lsa_samples.json): lo que la persona vio en la pantalla.
                     No tiene referencia estándar justa: ya entrenó con la sesión 1.

Además mide la detección de manos (HandLandmarker) por condición, que explica buena
parte de la caída sin pasar por el modelo. Aplica las correcciones de la grabación
(`resultados_captura_E3/correcciones_sesion2.json`).

Escribe en resultados_robustez/<fecha>_<etiqueta>/ (resultados.json, resumen.md,
robustez.tex y robustez.png). No toca models/ ni la app.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH, ver CLAUDE.md):
    python scripts/robustez.py --semillas 0 1 2 3 4 --etiqueta p1p2p3
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validar import entrenar

from lsa import dataset as ds
from lsa import dataset_v2
from lsa import robustez as rb

RAIZ = ds.RAIZ
CORRECCIONES = os.path.join(RAIZ, "resultados_captura_E3", "correcciones_sesion2.json")
MODELO_APP = os.path.join(RAIZ, "runs", "2026-10-16_final", "actions_15_opt.tflite")
CALIBRACION_APP = os.path.join(os.path.dirname(os.path.dirname(RAIZ)), "TraductorLSA", "app", "src", "main",
                               "assets", "calibracion.json")
ESQUEMAS = ["persona_nueva", "persona_conocida", "app"]


def condiciones(d, correcciones):
    return np.array([rb.condicion_efectiva(p, int(o), b, c, correcciones)
                     for p, o, b, c in zip(d.persona, d.orden, d.bloque, d.condicion, strict=True)])


def ids(d):
    return np.array([f"{p}-{o}" for p, o in zip(d.persona, d.orden, strict=True)])


def esquema_entrenado(esquema, d, b, cond_b, semillas, epocas):
    """LOPO o 'menos la tanda 2': el mismo modelo evalúa la referencia y las condiciones."""
    nada = d.clases.index(ds.NADA) if ds.NADA in d.clases else -1
    C = len(d.clases)
    preds = []
    for semilla in semillas:
        for p in sorted(set(b.persona.tolist())):
            if esquema == "persona_nueva":
                tr = d.persona != p
                ref = (d.persona == p) & (d.fuente == "telefono") & (d.y != nada)
            else:
                tr = d.sesion != f"{p}-A2"
                ref = (d.sesion == f"{p}-A2") & (d.y != nada)
            prueba = b.persona == p
            t0 = time.time()
            modelo = entrenar(d.X[tr], d.y[tr], C, semilla, epocas)
            X = np.concatenate([d.X[ref], b.X[prueba]])
            probs = modelo.predict(X, verbose=0)
            filas = [(i, rb.ESTANDAR, d) for i in np.where(ref)[0]] + \
                    [(i, cond_b[i], b) for i in np.where(prueba)[0]]
            for (i, cond, origen), pr in zip(filas, probs, strict=True):
                preds.append({"id": f"{origen.persona[i]}-{origen.orden[i]}", "persona": p,
                              "dispositivo": str(origen.dispositivo[i]), "condicion": cond,
                              "clase": d.clases[origen.y[i]], "y": int(origen.y[i]), "pred": int(pr.argmax()),
                              "conf": float(pr.max()), "semilla": semilla})
            acc_ref = np.mean([q["y"] == q["pred"] for q in preds[-len(filas):][:ref.sum()]])
            acc_b = np.mean([q["y"] == q["pred"] for q in preds[-len(filas):][ref.sum():]])
            print(f"  {esquema} semilla {semilla} {p}: estándar {acc_ref:.3f} (n={ref.sum()}), "
                  f"sesión 2 {acc_b:.3f} (n={prueba.sum()}) [{time.time() - t0:.0f} s]", flush=True)
    return preds


def secuencias_en_vivo(clases):
    """Por toma de la sesión 2, la secuencia v1 que la app clasificó en vivo.

    lsa_samples.json no guarda el orden de la planilla: se empareja con el v2 por
    posición dentro del día de la sesión 2, y solo si las etiquetas coinciden una a una
    (las tomas deshechas o borradas se quitaron de los dos archivos)."""
    import glob
    out = {}
    for ruta in sorted(glob.glob(os.path.join(RAIZ, "data", "captura_E3", "*", "lsa_samples_v2.jsonl"))):
        carpeta = os.path.dirname(ruta)
        b = [m for m in dataset_v2.leer(ruta) if m.get("sesion", {}).get("bloque") == "B"]
        if not b:
            continue
        fecha = date.fromtimestamp(b[0]["ts"] / 1000).isoformat()
        with open(os.path.join(carpeta, "lsa_samples.json"), encoding="utf-8") as f:
            v1 = json.load(f)["by_date"].get(fecha, [])
        # Ese día puede tener tomas de otra cosa antes (pruebas): las de la sesión 2 son las últimas.
        v1 = v1[-len(b):]
        if [m["label"] for m in v1] != [m["label"] for m in b]:
            print(f"  aviso: {carpeta}: v1 y v2 no coinciden en la sesión 2, se omite en 'app'", flush=True)
            continue
        for m1, m2 in zip(v1, b, strict=True):
            out[f"{m2['sesion']['codigo']}-{m2['sesion']['orden']}"] = np.asarray(m1["seq"], np.float32)
    return out


def esquema_app(b, cond_b):
    from lsa.export import inferir_tflite
    with open(MODELO_APP, "rb") as f:
        tfl = f.read()
    with open(CALIBRACION_APP, encoding="utf-8") as f:
        cal = json.load(f)
    sha = hashlib.sha256(tfl).hexdigest()
    if sha != cal.get("modelo_sha256"):
        raise SystemExit(f"El modelo {MODELO_APP} ({sha[:8]}) no es el de la calibración de la app")
    with open(os.path.join(os.path.dirname(MODELO_APP), "words.json"), encoding="utf-8") as f:
        clases_app = json.load(f)["word_ids"]
    if clases_app != b.clases:
        raise SystemExit("words.json del modelo de la app no coincide con las clases del análisis")
    vivo = secuencias_en_vivo(b.clases)
    sel = [i for i, k in enumerate(ids(b)) if k in vivo]
    X = np.stack([vivo[ids(b)[i]] for i in sel])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibrar import con_temperatura
    probs = con_temperatura(inferir_tflite(tfl, X), cal["temperatura"])
    nada = b.clases.index(ds.NADA)
    preds = []
    for i, pr in zip(sel, probs, strict=True):
        pred, conf = int(pr.argmax()), float(pr.max())
        preds.append({"id": ids(b)[i], "persona": str(b.persona[i]), "dispositivo": str(b.dispositivo[i]),
                      "condicion": cond_b[i], "clase": b.clases[b.y[i]], "y": int(b.y[i]), "pred": pred,
                      "conf": conf, "semilla": 0, "mostrada": bool(conf >= cal["umbral"] and pred != nada)})
    return preds, {"temperatura": cal["temperatura"], "umbral": cal["umbral"], "modelo_sha256": sha}


def resumir(preds, clases):
    """Por condición (y por persona dentro de cada una): acierto medio entre semillas,
    IC de Wilson sobre las tomas y caída respecto de la referencia estándar."""
    por_muestra = rb.acierto_por_muestra(preds)
    info = {p["id"]: p for p in preds}
    conds = [c for c in rb.CONDICIONES if any(p["condicion"] == c for p in preds)]
    ref = np.array([v for k, v in por_muestra.items() if info[k]["condicion"] == rb.ESTANDAR])
    out = {"condiciones": {}}
    for c in conds:
        ks = [k for k in por_muestra if info[k]["condicion"] == c]
        v = np.array([por_muestra[k] for k in ks])
        semillas = sorted({p["semilla"] for p in preds})
        por_sem = [np.mean([p["y"] == p["pred"] for p in preds if p["condicion"] == c and p["semilla"] == s])
                   for s in semillas]
        r = {"n": len(ks), "acierto": float(v.mean()), "ic95": rb.wilson(v.sum(), len(v)),
             "desvio_semillas": float(np.std(por_sem, ddof=1)) if len(por_sem) > 1 else 0.0, "por_persona": {}}
        if c != rb.ESTANDAR and len(ref):
            r["caida_vs_estandar"] = rb.diferencia_bootstrap(ref, v)
        for persona in sorted({info[k]["persona"] for k in ks}):
            vp = np.array([por_muestra[k] for k in ks if info[k]["persona"] == persona])
            r["por_persona"][persona] = {"n": len(vp), "acierto": float(vp.mean()),
                                         "dispositivo": next(info[k]["dispositivo"] for k in ks
                                                             if info[k]["persona"] == persona)}
        # Con qué se confunde: la predicción más frecuente cuando se equivoca.
        errores = [clases[p["pred"]] for p in preds if p["condicion"] == c and p["pred"] != p["y"]]
        r["confusion_mas_comun"] = max(set(errores), key=errores.count) if errores else None
        if any("mostrada" in p for p in preds):
            m = [p for p in preds if p["condicion"] == c]
            mostradas = [p for p in m if p["mostrada"]]
            r["mostradas"] = len(mostradas) / len(m)
            r["acierto_mostradas"] = (float(np.mean([p["y"] == p["pred"] for p in mostradas]))
                                      if mostradas else None)
        out["condiciones"][c] = r
    return out


def deteccion_por_condicion(correcciones):
    """Lo que vio el extractor en cada condición, desde los crudos (sesión 1 = estándar)."""
    import glob
    filas = {}
    for ruta in sorted(glob.glob(ds.TELEFONO)):
        for m in dataset_v2.leer(ruta):
            s = m.get("sesion") or {}
            if s.get("codigo") in (None, "PRUEBA") or s.get("bloque") not in ("A", "B"):
                continue
            c = rb.condicion_efectiva(s["codigo"], int(s["orden"]), s["bloque"], s["condicion"], correcciones)
            det = rb.deteccion(m)
            det["dispositivo"] = m["dispositivo"]
            filas.setdefault(c, []).append(det)
    out = {}
    for c in [c for c in rb.CONDICIONES if c in filas]:
        f = filas[c]
        ambas = [x["ambas"] for x in f if x["ambas"] is not None]
        out[c] = {"tomas": len(f), "frames_con_mano": float(np.mean([x["con_mano"] for x in f])),
                  "ambas_en_senas_de_dos_manos": float(np.mean(ambas)) if ambas else None,
                  "fps_mediana": float(np.median([x["fps"] for x in f]))}
    return out


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}".replace(".", ",")


def puntos(v):
    return f"{100 * v:+.1f}".replace(".", ",")


def tabla_md(res, esquema, titulo, que):
    r = res["esquemas"][esquema]["condiciones"]
    personas = sorted({p for c in r.values() for p in c["por_persona"]})
    md = [f"### {titulo}", "", que, "",
          "| Condición | Tomas | Acierto (%) | IC 95 % | Caída vs estándar (pts, IC 95 %) | "
          + " | ".join(personas) + " | Confusión más común |",
          "|---|---|---|---|---|" + "---|" * len(personas) + "---|"]
    for c, v in r.items():
        caida = v.get("caida_vs_estandar")
        txt_caida = (f"{puntos(caida[0])} ({puntos(caida[1])} a {puntos(caida[2])})" if caida else "referencia")
        pp = " | ".join(pct(v["por_persona"][p]["acierto"]) if p in v["por_persona"] else "—" for p in personas)
        md.append(f"| {c} | {v['n']} | {pct(v['acierto'])} ± {pct(v['desvio_semillas'])} | "
                  f"{pct(v['ic95'][0])}–{pct(v['ic95'][1])} | {txt_caida} | {pp} | {v['confusion_mas_comun'] or '—'} |")
    return md


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--esquemas", nargs="+", default=ESQUEMAS, choices=ESQUEMAS)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--etiqueta", default="run")
    args = ap.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    salida = os.path.join(RAIZ, "resultados_robustez", f"{date.today().isoformat()}_{args.etiqueta}")
    os.makedirs(salida, exist_ok=True)
    correcciones = rb.leer_correcciones(CORRECCIONES)
    d = ds.cargar(incluir_nada=True)
    b = ds.cargar(incluir_nada=True, bloques=("B",))
    cond_b = condiciones(b, correcciones)
    print(d.resumen(), "\nsesión 2:", b.resumen(), flush=True)
    res = {"datos": d.resumen(), "sesion2": b.resumen(), "semillas": args.semillas, "clases": d.clases,
           "correcciones": {k: [list(t) for t in v] for k, v in correcciones.items()},
           "deteccion": deteccion_por_condicion(correcciones), "esquemas": {}, "predicciones": {}}
    with open(CORRECCIONES, encoding="utf-8") as f:
        res["intentos_cortados"] = {p: c["intentos_cortados_aprox"] for p, c in json.load(f).items()
                                    if isinstance(c, dict) and "intentos_cortados_aprox" in c}

    for esquema in args.esquemas:
        t0 = time.time()
        if esquema == "app":
            preds, meta = esquema_app(b, cond_b)
            res["app"] = meta
        else:
            preds = esquema_entrenado(esquema, d, b, cond_b, args.semillas, args.epocas)
        res["esquemas"][esquema] = resumir(preds, d.clases)
        res["predicciones"][esquema] = preds
        print(f"{esquema}: listo ({time.time() - t0:.0f} s)", flush=True)
        with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)

    escribir_resumen(res, salida)
    print(f"-> {salida}")


def escribir_resumen(res, salida):
    md = ["# Robustez por condición (sesión 2)", "",
          f"Generado por `scripts/robustez.py`. No editar a mano. Semillas: {res['semillas']}.", "",
          f"Entrenamiento: {res['datos']}.", "", f"Sesión 2: {res['sesion2']}.", "",
          "Cada condición se mide con el mismo modelo que su referencia estándar. El IC 95 % es de Wilson "
          "sobre las tomas (cada toma promedia sus semillas); la caída tiene IC por bootstrap. Con 12 tomas "
          "por persona y condición los intervalos son anchos: las diferencias chicas no son concluyentes.", ""]
    if res["correcciones"]:
        md += ["Correcciones aplicadas (de `correcciones_sesion2.json`): " + "; ".join(
            f"{p} órdenes {a}–{z} → {rb.nombre_condicion(c)}" for p, v in res["correcciones"].items()
            for a, z, c in v) + ". P3 no tiene «girado 30°» (descartada en la grabación) y a P2 le falta "
            "«por favor» a 30°.", ""]
    md += ["## Reconocimiento", ""]
    textos = {"persona_nueva": ("Persona no vista (LOPO)", "Entrena sin la persona evaluada. Estándar = sus tomas "
                                "de la sesión 1."),
              "persona_conocida": ("Persona conocida, sesión nueva", "Entrena con todo menos la tanda 2 de la "
                                   "persona. Estándar = esa tanda 2.")}
    for e, (t, q) in textos.items():
        if e in res["esquemas"]:
            md += tabla_md(res, e, t, q) + [""]
    if "app" in res["esquemas"]:
        r = res["esquemas"]["app"]["condiciones"]
        a = res["app"]
        md += ["### La app en vivo", "",
               f"Modelo publicado (sha `{a['modelo_sha256'][:8]}`), temperatura {a['temperatura']:.2f}, umbral "
               f"{a['umbral']}, sobre la secuencia que la app calculó en cada toma. Ya entrenó con la sesión 1 de "
               "estas personas, así que no hay referencia estándar justa.", "",
               "| Condición | Tomas | Acierto (%) | Se muestra (%) | Acierto de lo mostrado (%) |", "|---|---|---|---|---|"]
        for c, v in r.items():
            md.append(f"| {c} | {v['n']} | {pct(v['acierto'])} | {pct(v['mostradas'])} | {pct(v['acierto_mostradas'])} |")
        md.append("")
    md += ["## Detección de manos (HandLandmarker)", "",
           "| Condición | Tomas | Frames con mano (%) | Ambas manos en «por favor» y «cómo estás» (%) | FPS (mediana) |",
           "|---|---|---|---|---|"]
    for c, v in res["deteccion"].items():
        md.append(f"| {c} | {v['tomas']} | {pct(v['frames_con_mano'])} | {pct(v['ambas_en_senas_de_dos_manos'])} | "
                  f"{v['fps_mediana']:.1f} |".replace(".", ","))
    md += ["", "Ojo al leer por condición: no todas tienen a las tres personas (ver las columnas por persona), "
           "así que una condición mezcla su efecto con el de quién y con qué dispositivo la grabó. Lo mismo "
           "pasa con los FPS: la mediana sube donde pesa más la tablet (16-18 FPS) y baja donde pesa el A55 (6 FPS)."]
    md += ["", "Los frames de una toma guardada siempre tienen alguna mano (la captura se corta si se pierde): "
           "la degradación del extractor se ve en las dos manos y en los intentos cortados.", ""]
    if res.get("intentos_cortados"):
        conds = [c for c in rb.CONDICIONES if any(c in v for v in res["intentos_cortados"].values())]
        md += ["## Intentos cortados durante la grabación (aprox.)", "",
               "| Persona | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
        for p, v in sorted(res["intentos_cortados"].items()):
            md.append(f"| {p} | " + " | ".join(v.get(c, "—") for c in conds) + " |")
        md += ["", "Fuente: seguimiento en vivo y métricas de cada dispositivo (ver `correcciones_sesion2.json`).", ""]

    tex = tabla_tex(res)
    md += ["## Tabla en LaTeX (booktabs)", "", "```latex", *tex, "```", ""]
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    with open(os.path.join(salida, "robustez.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tex) + "\n")
    grafico(res, os.path.join(salida, "robustez.png"))
    print("\n".join(md))


def tabla_tex(res):
    esq = [e for e in ("persona_nueva", "persona_conocida") if e in res["esquemas"]]
    nombres = {"persona_nueva": "Persona no vista", "persona_conocida": "Persona conocida"}
    tex = [r"\begin{tabular}{l" + "rr" * len(esq) + "r}", r"\toprule",
           "Condición & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{nombres[e]}}}" for e in esq)
           + r" & Ambas manos (\%) \\",
           " & " + " & ".join(r"Acierto (\%) & Caída (pts)" for _ in esq) + r" & \\", r"\midrule"]
    for c in rb.CONDICIONES:
        if not any(c in res["esquemas"][e]["condiciones"] for e in esq):
            continue
        celdas = []
        for e in esq:
            v = res["esquemas"][e]["condiciones"].get(c)
            if v is None:
                celdas += ["—", "—"]
                continue
            caida = v.get("caida_vs_estandar")
            celdas += [pct(v["acierto"]), puntos(caida[0]) if caida else "ref."]
        det = res["deteccion"].get(c, {}).get("ambas_en_senas_de_dos_manos")
        tex.append(f"{c} & " + " & ".join(celdas) + f" & {pct(det)}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    return tex


def grafico(res, ruta):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    esq = [e for e in ("persona_nueva", "persona_conocida") if e in res["esquemas"]]
    if not esq:
        return
    conds = [c for c in rb.CONDICIONES if any(c in res["esquemas"][e]["condiciones"] for e in esq)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ancho = 0.8 / len(esq)
    nombres = {"persona_nueva": "Persona no vista", "persona_conocida": "Persona conocida"}
    for k, e in enumerate(esq):
        r = res["esquemas"][e]["condiciones"]
        xs = [i + (k - (len(esq) - 1) / 2) * ancho for i, c in enumerate(conds) if c in r]
        ys = [100 * r[c]["acierto"] for c in conds if c in r]
        err = np.array([[100 * (r[c]["acierto"] - r[c]["ic95"][0]), 100 * (r[c]["ic95"][1] - r[c]["acierto"])]
                        for c in conds if c in r]).T
        ax.bar(xs, ys, ancho, yerr=err, capsize=3, label=nombres[e])
    ax.set_xticks(range(len(conds)), conds, rotation=15)
    ax.set_ylabel("Acierto (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")
    ax.set_title("Reconocimiento por condición (IC 95 %)")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)


if __name__ == "__main__":
    main()
