"""Métrica de consistencia estadística σ de la propuesta (PY-E7), sobre las tomas reales.

σ mide si entradas parecidas reciben salidas parecidas, sin usar etiquetas
(`lsa/consistencia.py` tiene la definición de la propuesta y las decisiones que deja
abiertas). Acá se calcula:

  1. Con el modelo publicado en la app (con su temperatura): en la sesión 1 y en cada
     condición de la sesión 2.
  2. Con modelos que no vieron a la persona (LOPO, como en robustez.py): σ sobre las
     tomas de esa persona, la situación de alguien nuevo.
  3. Si σ_i sirve de alarma sin etiquetas: σ_i medio de las tomas que el modelo acierta
     contra las que erra.

ε sale de los datos: percentiles de las distancias entre pares de tomas de la sesión 1
del teléfono, y el mismo ε se usa en todos los subconjuntos. Se informan varios.

Escribe en resultados_consistencia/<fecha>_<etiqueta>/ (resultados.json, resumen.md,
consistencia.tex). No toca models/ ni la app.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH):
    python scripts/consistencia.py --semillas 0 1 2 --etiqueta p1p2p3
"""
import argparse
import json
import os
import sys
from datetime import date

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustez import CALIBRACION_APP, CORRECCIONES, MODELO_APP, condiciones
from validar import entrenar

from lsa import consistencia as cs
from lsa import dataset as ds
from lsa import robustez as rb

RAIZ = ds.RAIZ
PERCENTILES = [0.5, 1, 2, 5, 10]
PUREZA_MINIMA = 0.9  # ε principal: el más grande con ≥ 90 % de vecinos de la misma seña


def pct(v):
    return "—" if v is None else f"{100 * v:.1f}".replace(".", ",")


def num(v, d=3):
    return "—" if v is None else f"{v:.{d}f}".replace(".", ",")


def resumir(P, y, D, epsilons, mascara):
    """σ de las tomas de [mascara], con las vecindades buscadas en TODAS las tomas.

    En la sesión 2 cada persona hizo cada seña una sola vez por condición: buscando
    vecinos solo dentro de la condición, casi ninguna toma tenía un vecino de la misma
    seña. Con todas las tomas, σ_i de una toma girada 30° compara su salida con la de las
    tomas parecidas (por ejemplo, la misma seña de frente), que es lo que la propuesta
    quiere medir: si una variación chica de la entrada cambia la respuesta."""
    ok = P.argmax(1) == y
    out = {"n": int(mascara.sum()), "accuracy": float(np.mean(ok[mascara])), "por_epsilon": {}}
    for p, e in epsilons.items():
        r = cs.sigma(P, D, e)
        en = r["en_I"] & mascara
        tamanos = (D < e).sum(axis=1)
        sigmas_i = r["sigmas_i"]

        def media(m, sigmas_i=sigmas_i):
            return float(np.mean(sigmas_i[m])) if m.any() else None
        out["por_epsilon"][str(p)] = {
            "sigma": media(en), "n_I": int(en.sum()),
            "vecinos_medio": float(np.mean(tamanos[en])) if en.any() else 0.0,
            "sigma_i_aciertos": media(en & ok), "sigma_i_errores": media(en & ~ok),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--etiqueta", default="run")
    args = ap.parse_args()

    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from calibrar import con_temperatura

    from lsa.export import inferir_tflite

    d = ds.cargar(incluir_nada=True)
    b = ds.cargar(incluir_nada=True, bloques=("B",))
    correcciones = rb.leer_correcciones(CORRECCIONES)
    cond_b = condiciones(b, correcciones)
    tel = np.where(d.fuente == "telefono")[0]
    # Evaluación: tomas del teléfono, sesión 1 + sesión 2.
    X = np.concatenate([d.X[tel], b.X])
    y = np.concatenate([d.y[tel], b.y])
    persona = np.concatenate([d.persona[tel], b.persona])
    cond = np.concatenate([np.array([rb.ESTANDAR] * len(tel)), cond_b])
    sesion = np.concatenate([np.ones(len(tel), int), np.full(len(b.y), 2)])
    D = cs.distancias(X)
    s1 = np.where(sesion == 1)[0]
    epsilons = {p: cs.epsilon_por_percentil(D[np.ix_(s1, s1)], p) for p in PERCENTILES}
    purezas = {p: cs.pureza(D[np.ix_(s1, s1)], y[s1], e) for p, e in epsilons.items()}
    validos = [p for p in PERCENTILES if purezas[p] is not None and purezas[p] >= PUREZA_MINIMA]
    principal = max(validos) if validos else min(PERCENTILES)
    print(f"{len(y)} tomas del teléfono (sesión 1: {len(s1)}, sesión 2: {len(b.y)}); ε por percentil: "
          + ", ".join(f"p{p}={e:.2f} (pureza {purezas[p]:.2f})" for p, e in epsilons.items())
          + f"; principal p{principal}", flush=True)

    # 1. Modelo publicado (lo que corre en la app).
    with open(MODELO_APP, "rb") as f:
        tfl = f.read()
    with open(CALIBRACION_APP, encoding="utf-8") as f:
        T = json.load(f)["temperatura"]
    P_app = con_temperatura(inferir_tflite(tfl, X), T)
    res = {"n": len(y), "epsilons": {str(k): v for k, v in epsilons.items()}, "principal": principal,
           "purezas": {str(k): v for k, v in purezas.items()}, "pureza_minima": PUREZA_MINIMA,
           "semillas": args.semillas, "app": {}, "persona_nueva": {}}
    res["app"]["sesion1"] = resumir(P_app, y, D, epsilons, sesion == 1)
    for c in [c for c in rb.CONDICIONES if c != rb.ESTANDAR and (cond == c).any()]:
        res["app"][c] = resumir(P_app, y, D, epsilons, (sesion == 2) & (cond == c))

    # 2. Persona no vista: para cada persona, modelo sin ella; σ sobre sus tomas.
    C = len(d.clases)
    por_semilla = []
    for semilla in args.semillas:
        P_lopo = np.zeros_like(P_app)
        for p in sorted(set(persona.tolist())):
            modelo = entrenar(d.X[d.persona != p], d.y[d.persona != p], C, semilla, args.epocas)
            m = persona == p
            P_lopo[m] = modelo.predict(X[m], verbose=0)
            print(f"  semilla {semilla}: modelo sin {p} listo", flush=True)
        fila = {"sesion1": resumir(P_lopo, y, D, epsilons, sesion == 1)}
        for c in [c for c in rb.CONDICIONES if c != rb.ESTANDAR and (cond == c).any()]:
            fila[c] = resumir(P_lopo, y, D, epsilons, (sesion == 2) & (cond == c))
        por_semilla.append(fila)
    for clave in por_semilla[0]:
        sig = [f[clave]["por_epsilon"][str(principal)]["sigma"] for f in por_semilla]
        sig = [s for s in sig if s is not None]
        res["persona_nueva"][clave] = {
            "accuracy": float(np.mean([f[clave]["accuracy"] for f in por_semilla])),
            "sigma_media": float(np.mean(sig)) if sig else None,
            "sigma_desvio": float(np.std(sig, ddof=1)) if len(sig) > 1 else 0.0,
            "por_semilla": [f[clave] for f in por_semilla]}

    salida = os.path.join(RAIZ, "resultados_consistencia", f"{date.today().isoformat()}_{args.etiqueta}")
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)
    escribir_resumen(res, salida)


def escribir_resumen(res, salida):
    e = res["epsilons"]
    pr_ = str(res["principal"])
    md = ["# Consistencia estadística σ (propuesta, sección de evaluación)", "",
          "Generado por `scripts/consistencia.py`. No editar a mano.", "",
          f"{res['n']} tomas del teléfono. σ_i = promedio sobre las clases de la desviación estándar sesgada de "
          "las probabilidades de los vecinos (N_i incluye a la muestra; los vecinos se buscan en todas las tomas, "
          "de las dos sesiones); σ de un grupo = promedio de σ_i de sus tomas con al menos otro vecino. ε = percentil de las distancias entre pares de la sesión 1: "
          + ", ".join(f"p{k} = {num(v, 2)}" for k, v in e.items()) + f". Tabla principal con p{pr_}: el ε más "
          f"grande con al menos {pct(res['pureza_minima'])} % de vecinos de la misma seña (con ε más grande entran "
          "señas distintas y σ mide cuánto se mezclan las clases, no la estabilidad).", "",
          "Ojo: un modelo que reparte la probabilidad por igual entre todas las clases tiene σ bajo sin acertar. "
          "σ se lee siempre junto al acierto.", "",
          "σ va de 0 (vecinos con las mismas probabilidades) a 0,5 (vecinos que se reparten entre dos clases con "
          "certeza); más bajo es más consistente.", "",
          "## Modelo publicado en la app (temperatura incluida)", "",
          "| Datos | Tomas | Acierto (%) | σ | Con vecinos | σ_i aciertos | σ_i errores |",
          "|---|---|---|---|---|---|---|"]
    for clave, v in res["app"].items():
        q = v["por_epsilon"][pr_]
        nombre = "sesión 1" if clave == "sesion1" else clave
        md.append(f"| {nombre} | {v['n']} | {pct(v['accuracy'])} | {num(q['sigma'])} "
                  f"| {q['n_I']} | {num(q['sigma_i_aciertos'])} | {num(q['sigma_i_errores'])} |")
    md += ["", "La app ya vio la sesión 1 de estas personas: su σ ahí es optimista.", "",
           "## Persona no vista (cada persona con un modelo entrenado sin ella)", "",
           f"Semillas: {res['semillas']}. σ con p{pr_}, media ± desvío entre semillas.", "",
           "| Datos | Acierto (%) | σ |", "|---|---|---|"]
    for clave, v in res["persona_nueva"].items():
        md.append(f"| {'sesión 1' if clave == 'sesion1' else clave} | {pct(v['accuracy'])} | "
                  f"{num(v['sigma_media'])} ± {num(v['sigma_desvio'])} |")
    md += ["", "## Sensibilidad a ε (modelo publicado, sesión 1)", "",
           "| ε | Vecinos de la misma seña (%) | σ | Con vecinos | Vecinos por toma |", "|---|---|---|---|---|"]
    for k, q in res["app"]["sesion1"]["por_epsilon"].items():
        md.append(f"| p{k} = {num(e[k], 2)} | {pct(res['purezas'][k])} | {num(q['sigma'])} | {q['n_I']} | "
                  f"{num(q['vecinos_medio'], 1)} |")
    tex = [r"\begin{tabular}{lrrrr}", r"\toprule",
           r"Datos & Acierto app (\%) & $\sigma$ app & Acierto nueva (\%) & $\sigma$ nueva \\", r"\midrule"]
    for clave, v in res["app"].items():
        q = v["por_epsilon"][pr_]
        n = res["persona_nueva"].get(clave, {})
        nombre = "sesión 1" if clave == "sesion1" else clave
        tex.append(f"{nombre} & {pct(v['accuracy'])} & {num(q['sigma'])} & {pct(n.get('accuracy'))} & "
                   f"{num(n.get('sigma_media'))}" + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    md += ["", "## Tabla en LaTeX (booktabs)", "", "```latex", *tex, "```", ""]
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    with open(os.path.join(salida, "consistencia.tex"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(tex) + "\n")
    print("\n".join(md))
    print(f"-> {salida}")


if __name__ == "__main__":
    main()
