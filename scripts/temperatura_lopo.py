"""¿Con qué datos se ajusta la temperatura de calibración y cuánto importa? (auditoría, ML-M7)

`calibrar.py` ajusta, en cada pliegue LOPO (persona p afuera), la temperatura T sobre un
20 % reservado de los datos de entrenamiento de ese pliegue. Ese 20 % no lo vio el modelo,
y tampoco tiene a p, pero sí tiene a las personas con las que se entrenó (y, salvo cuando
p = P1, está dominado por la webcam de P1). La app, en cambio, ve personas nuevas. Acá se
comparan, sobre las mismas predicciones de personas no vistas, varias formas de elegir T:

  sin              T = 1 (sin calibrar).
  publicada        la T fija que usa la app (calibracion_app.json, solo se lee).
  receta_mediana   la mediana de las T de los pliegues de esa corrida (lo que publica calibrar.py).
  receta_pliegue   la T del propio pliegue (set de calibración del 20 %).
  cal_telefono     como la receta, pero ajustada solo con las tomas de teléfono del 20 %
                   (solo si se reentrena: hacen falta esas predicciones).
  anidada          ajustada con las predicciones LOPO de las OTRAS personas, en la misma
                   semilla: cada una vino de un modelo que no la vio. Es la versión honesta
                   de «calibrar sobre personas nuevas».
  oraculo          ajustada sobre la misma persona que se mide. No es desplegable: es la
                   cota de lo mejor que puede dar una sola T.

Para cada una: ECE (15 bins), log-loss, y con el umbral de la app la fracción de señas
mostradas y el acierto entre lo mostrado ("nada" no se muestra), por persona.

Modos (se pueden combinar):
  --desde <carpeta>   reanaliza una corrida de calibrar.py que guardó predicciones.npz
                      (las T de cada pliegue salen de su resultados.json). No entrena.
  --entrenar          repite la receta de calibrar.py (mismas semillas, LOPO, 20 % de
                      calibración, mismo `entrenar`) y guarda también las predicciones del
                      set de calibración, en <salida>/entrenada/.

No escribe calibracion_app.json, no toca models/ ni la app. Solo escribe en --salida, que
tiene que ser una carpeta nueva.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH, ver CLAUDE.md):
    python scripts/temperatura_lopo.py --desde resultados_calibracion/2026-11-10_lopo \\
        --entrenar --salida resultados_calibracion/2026-11-14_temperatura
    # rehacer el análisis sin volver a entrenar (la salida de --entrenar también sirve de --desde):
    python scripts/temperatura_lopo.py --desde resultados_calibracion/2026-11-10_lopo \\
        resultados_calibracion/2026-11-14_temperatura/entrenada --salida <carpeta nueva>
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrar import ajustar_temperatura, cobertura, con_temperatura, ece, log_loss

from lsa import dataset as ds


def _pliegues(semilla, persona):
    return [(int(s), str(q)) for s in sorted(set(semilla.tolist())) for q in sorted(set(persona.tolist()))]


def temperatura_anidada(p, y, persona, semilla, ajustar=ajustar_temperatura):
    """Por pliegue (semilla, persona q): T ajustada con las predicciones de las otras
    personas en la misma semilla. Ninguna de esas predicciones es de q ni de un modelo
    que haya visto a la persona predicha."""
    return {(s, q): ajustar(p[(semilla == s) & (persona != q)], y[(semilla == s) & (persona != q)])
            for s, q in _pliegues(semilla, persona)}


def temperatura_oraculo(p, y, persona, semilla, ajustar=ajustar_temperatura):
    """Por pliegue: T ajustada sobre las mismas predicciones que se evalúan (cota, no desplegable)."""
    return {(s, q): ajustar(p[(semilla == s) & (persona == q)], y[(semilla == s) & (persona == q)])
            for s, q in _pliegues(semilla, persona)}


def aplicar_por_pliegue(p, persona, semilla, temperaturas):
    """Aplica a cada fila la T de su pliegue. `temperaturas`: {(semilla, persona): T} o un número."""
    out = np.empty_like(p, dtype=np.float64)
    for s, q in _pliegues(semilla, persona):
        m = (semilla == s) & (persona == q)
        t = temperaturas if np.isscalar(temperaturas) else temperaturas[(s, q)]
        out[m] = con_temperatura(p[m], t)
    return out


def metricas(p, y, nada, umbral, bins=15):
    """Calibración y uso en la app de un conjunto de predicciones: ECE, log-loss, fracción
    mostrada con el umbral (predicción distinta de "nada" y confianza ≥ umbral) y acierto
    de lo mostrado."""
    conf, pred = p.max(1), p.argmax(1)
    mostrada = (conf >= umbral) & (pred != nada)
    return {"n": len(y), "accuracy": float((pred == y).mean()), "ece": ece(p, y, bins),
            "logloss": log_loss(p, y), "mostradas": float(mostrada.mean()),
            "n_mostradas": int(mostrada.sum()), "confianza_media": float(conf.mean()),
            # Lo que la app muestra como «xx %», para comparar con lo que de verdad acierta.
            "confianza_media_mostradas": float(conf[mostrada].mean()) if mostrada.any() else None,
            "acierto_mostradas": float((pred[mostrada] == y[mostrada]).mean()) if mostrada.any() else None}


def por_persona(p, y, persona, semilla, nada, umbral):
    """Métricas por persona (juntando las semillas) y, para ver la variación, el rango
    entre semillas de la fracción mostrada y del acierto de lo mostrado."""
    out = {}
    for q in sorted(set(persona.tolist())):
        m = persona == q
        r = metricas(p[m], y[m], nada, umbral)
        por_sem = [metricas(p[m & (semilla == s)], y[m & (semilla == s)], nada, umbral)
                   for s in sorted(set(semilla[m].tolist()))]
        r["mostradas_rango_semillas"] = [min(x["mostradas"] for x in por_sem), max(x["mostradas"] for x in por_sem)]
        a = [x["acierto_mostradas"] for x in por_sem if x["acierto_mostradas"] is not None]
        r["acierto_mostradas_rango_semillas"] = [min(a), max(a)] if a else None
        out[q] = r
    out["todas"] = metricas(p, y, nada, umbral)
    return out


def comparar(p, y, persona, semilla, nada, umbral, esquemas):
    """`esquemas`: {nombre: T fija o {(semilla, persona): T}}. Devuelve las métricas por
    persona de cada esquema y las T usadas."""
    out = {}
    for nombre, temps in esquemas.items():
        pt = aplicar_por_pliegue(p, persona, semilla, temps)
        out[nombre] = {"temperaturas": temps if np.isscalar(temps) else {f"{s}-{q}": t for (s, q), t in temps.items()},
                       "metricas": por_persona(pt, y, persona, semilla, nada, umbral),
                       # Con otra T, el mismo umbral significa otra cosa: la curva entera permite
                       # comparar a igual cobertura.
                       "curva_todas": cobertura(pt, y, nada)}
    return out


def composicion_calibracion(d, semillas, personas):
    """Qué hay en el 20 % de calibración de cada pliegue (mismo split que calibrar.py)."""
    from sklearn.model_selection import train_test_split
    out = {}
    for s in semillas:
        for q in personas:
            otros = np.where(d.persona != q)[0]
            _, cal = train_test_split(otros, test_size=0.2, random_state=s, stratify=d.y[otros])
            out[f"{s}-{q}"] = {"n": len(cal), "fuente": dict(Counter(d.fuente[cal].tolist())),
                               "persona": dict(sorted(Counter(d.persona[cal].tolist()).items()))}
    return out


def _cargar_corrida(carpeta):
    z = np.load(os.path.join(carpeta, "predicciones.npz"))
    for nombre in ("resultados.json", "pliegues.json"):
        ruta = os.path.join(carpeta, nombre)
        if os.path.exists(ruta):
            with open(ruta, encoding="utf-8") as f:
                j = json.load(f)
            pliegues = j["pliegues"] if isinstance(j, dict) else j
            break
    t_pliegue = {(int(f["semilla"]), str(f["persona"])): float(f["temperatura"]) for f in pliegues}
    extra = {}
    for f in pliegues:
        if "temperatura_cal_telefono" in f:
            extra.setdefault("cal_telefono", {})[(int(f["semilla"]), str(f["persona"]))] = f["temperatura_cal_telefono"]
    return {"p": z["p"].astype(np.float64), "y": z["y"], "persona": z["persona"], "semilla": z["semilla"],
            "clases": [str(c) for c in z["clases"]], "t_pliegue": t_pliegue, "extra": extra}


def _entrenar(d, semillas, salida):
    """La receta de calibrar.py, guardando además las predicciones del 20 % de calibración."""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from validar import entrenar
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print("GPU:", tf.config.list_physical_devices("GPU"), flush=True)
    tel = d.fuente == "telefono"
    personas = sorted(set(d.persona[tel].tolist()))
    P, Y, PER, SEM, pliegues = [], [], [], [], []
    PC, YC, FC, PERC, SEMC, QC = [], [], [], [], [], []
    for semilla in semillas:
        for q in personas:
            otros = np.where(d.persona != q)[0]
            tr, cal = train_test_split(otros, test_size=0.2, random_state=semilla, stratify=d.y[otros])
            modelo = entrenar(d.X[tr], d.y[tr], len(d.clases), semilla, 300)
            p_cal = modelo.predict(d.X[cal], verbose=0)
            t = ajustar_temperatura(p_cal, d.y[cal])
            ct = d.fuente[cal] == "telefono"
            t_tel = ajustar_temperatura(p_cal[ct], d.y[cal][ct])
            te = tel & (d.persona == q)
            P.append(modelo.predict(d.X[te], verbose=0))
            Y.append(d.y[te])
            PER.append(np.full(te.sum(), q))
            SEM.append(np.full(te.sum(), semilla))
            PC.append(p_cal)
            YC.append(d.y[cal])
            FC.append(d.fuente[cal])
            PERC.append(d.persona[cal])
            SEMC.append(np.full(len(cal), semilla))
            QC.append(np.full(len(cal), q))
            pliegues.append({"semilla": semilla, "persona": q, "temperatura": t, "temperatura_cal_telefono": t_tel,
                             "n_cal": len(cal), "n_cal_telefono": int(ct.sum())})
            print(f"  semilla {semilla} {q}: T={t:.3f} (solo teléfono {t_tel:.3f}, {int(ct.sum())} tomas)", flush=True)
            tf.keras.backend.clear_session()
    os.makedirs(salida, exist_ok=True)
    np.savez_compressed(os.path.join(salida, "predicciones.npz"), p=np.concatenate(P), y=np.concatenate(Y),
                        persona=np.concatenate(PER), semilla=np.concatenate(SEM), clases=np.array(d.clases),
                        p_cal=np.concatenate(PC), y_cal=np.concatenate(YC), fuente_cal=np.concatenate(FC),
                        persona_cal=np.concatenate(PERC), semilla_cal=np.concatenate(SEMC),
                        pliegue_cal=np.concatenate(QC))
    with open(os.path.join(salida, "pliegues.json"), "w", encoding="utf-8") as f:
        json.dump(pliegues, f, indent=2, ensure_ascii=False)


def analizar(c, t_publicada, umbral):
    nada = c["clases"].index(ds.NADA)
    p, y, persona, semilla = c["p"], c["y"], c["persona"], c["semilla"]
    esquemas = {"sin": 1.0, "publicada": t_publicada,
                "receta_mediana": float(np.median(list(c["t_pliegue"].values()))),
                "receta_pliegue": c["t_pliegue"], **c["extra"],
                "anidada": temperatura_anidada(p, y, persona, semilla),
                "oraculo": temperatura_oraculo(p, y, persona, semilla)}
    r = comparar(p, y, persona, semilla, nada, umbral, esquemas)
    # La T que se publicaría si se calibrara con predicciones de personas no vistas: una
    # sola, ajustada con todas las predicciones LOPO (evaluarla sobre ellas sería optimista;
    # para eso está la anidada).
    r["_t_lopo_todas"] = ajustar_temperatura(p, y)
    return r


def _pct(v):
    return "—" if v is None else f"{100 * v:.1f}".replace(".", ",")


def _num(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")


def resumen_md(resultados, t_publicada, umbral):
    md = ["# Temperatura de calibración: con qué datos se ajusta y cuánto importa (ML-M7)", "",
          f"Umbral de la app: {_num(umbral)}. T publicada en la app: {_num(t_publicada, 4)}. "
          "Generado por `scripts/temperatura_lopo.py`.", ""]
    for nombre, r in resultados["corridas"].items():
        a = r["analisis"]
        md += [f"## Corrida: {nombre}", "", r["descripcion"], ""]
        esquemas = [k for k in a if not k.startswith("_")]
        por_pliegue = [k for k in esquemas if isinstance(a[k]["temperaturas"], dict)]
        md += ["### Temperatura por pliegue (semilla-persona evaluada)", "",
               "| Pliegue | " + " | ".join(por_pliegue) + " |", "|---" * (len(por_pliegue) + 1) + "|"]
        for k in a[por_pliegue[0]]["temperaturas"]:
            md.append(f"| {k} | " + " | ".join(_num(a[e]["temperaturas"][k]) for e in por_pliegue) + " |")
        md.append("| mediana | " + " | ".join(_num(float(np.median(list(a[e]["temperaturas"].values()))))
                                          for e in por_pliegue) + " |")
        fijas = {e: a[e]["temperaturas"] for e in esquemas if e not in por_pliegue}
        md += ["", "Fijas: " + ", ".join(f"{e} = {_num(t, 3)}" for e, t in fijas.items())
               + f". Una sola T ajustada con todas las predicciones LOPO: {_num(a['_t_lopo_todas'], 3)}.", ""]
        md += ["### Efecto por persona (las semillas juntas)", "",
               "Mostradas y acierto de lo mostrado con el umbral de la app; entre paréntesis, el rango entre "
               "semillas. La argmax no cambia con T, así que el accuracy es el mismo en todos los esquemas.", "",
               "| Persona | Esquema | ECE % | Log-loss | Mostradas % | Acierto mostradas % | Confianza media "
               "de lo mostrado % |",
               "|---|---|---|---|---|---|---|"]
        personas = [q for q in a["sin"]["metricas"] if q != "todas"] + ["todas"]
        for q in personas:
            for e in esquemas:
                m = a[e]["metricas"][q]
                rm, ra = m.get("mostradas_rango_semillas"), m.get("acierto_mostradas_rango_semillas")
                rango_m = f" ({_pct(rm[0])}-{_pct(rm[1])})" if rm else ""
                rango_a = f" ({_pct(ra[0])}-{_pct(ra[1])})" if ra else ""
                md.append(f"| {q} (n={m['n']}) | {e} | {_pct(m['ece'])} | {_num(m['logloss'], 3)} | "
                          f"{_pct(m['mostradas'])}{rango_m} | {_pct(m['acierto_mostradas'])}{rango_a} "
                          f"| {_pct(m['confianza_media_mostradas'])} |")
        md.append("")
        if "composicion_calibracion" in r:
            md += ["### Qué hay en el 20 % de calibración de cada pliegue", "",
                   "| Pliegue | n | webcam | teléfono | por persona |", "|---|---|---|---|---|"]
            for k, v in r["composicion_calibracion"].items():
                md.append(f"| {k} | {v['n']} | {v['fuente'].get('webcam', 0)} | {v['fuente'].get('telefono', 0)} "
                          f"| {v['persona']} |")
            md.append("")
    return md


_DESCRIPCION_ENTRENADA = ("Misma receta que calibrar.py (LOPO, 20 % de calibración estratificado, mismo `entrenar`, "
                          "GPU), con las predicciones del set de calibración guardadas para ajustar también la T "
                          "solo con sus tomas de teléfono (`cal_telefono`).")


def _a_json(o):
    if isinstance(o, dict):
        return {str(k): _a_json(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_a_json(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desde", nargs="*", default=[], help="corridas de calibrar.py con predicciones.npz")
    ap.add_argument("--entrenar", action="store_true", help="repetir la receta y guardar el set de calibración")
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()
    salida = os.path.abspath(args.salida)
    if os.path.exists(os.path.join(salida, "resultados.json")):
        sys.exit(f"{salida} ya tiene resultados: usar una carpeta nueva.")

    with open(os.path.join(ds.RAIZ, "resultados_calibracion", "calibracion_app.json"), encoding="utf-8") as f:
        app = json.load(f)
    t_pub, umbral = float(app["temperatura"]), float(app["umbral"])

    d = ds.cargar(incluir_nada=True)
    personas = sorted(set(d.persona[d.fuente == "telefono"].tolist()))
    composicion = composicion_calibracion(d, args.semillas, personas)
    resultados = {"config": vars(args), "datos": d.resumen(), "t_publicada": t_pub, "umbral_app": umbral,
                  "corridas": {}}
    for carpeta in args.desde:
        c = _cargar_corrida(carpeta)
        ruta = os.path.relpath(carpeta, ds.RAIZ).replace(os.sep, "/")
        if "cal_telefono" in c["extra"]:  # una salida de --entrenar de este mismo script
            nombre, descripcion = f"{ruta} (reentrenada con la receta de calibrar.py)", _DESCRIPCION_ENTRENADA
        else:
            nombre = os.path.basename(os.path.normpath(carpeta)) + " (reanálisis, sin entrenar)"
            descripcion = f"Predicciones guardadas por calibrar.py en `{ruta}`. Las T de la receta salen de su " \
                          "resultados.json."
        resultados["corridas"][nombre] = {
            "descripcion": descripcion + f" {len(c['y'])} predicciones = {len(set(c['persona'].tolist()))} "
                           f"personas × {len(set(c['semilla'].tolist()))} semillas.",
            "analisis": analizar(c, t_pub, umbral), "composicion_calibracion": composicion}
    if args.entrenar:
        destino = os.path.join(salida, "entrenada")
        _entrenar(d, args.semillas, destino)
        c = _cargar_corrida(destino)
        resultados["corridas"]["2026-11-14 (reentrenada con la receta de calibrar.py)"] = {
            "descripcion": _DESCRIPCION_ENTRENADA + f" Semillas {args.semillas}.",
            "analisis": analizar(c, t_pub, umbral), "composicion_calibracion": composicion}

    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(_a_json(resultados), f, indent=2, ensure_ascii=False)
    md = resumen_md(_a_json(resultados), t_pub, umbral)
    with open(os.path.join(salida, "tablas.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print(f"-> {salida}")


if __name__ == "__main__":
    main()
