"""
Aprendizaje federado con las personas reales de la sesión de captura (PY-F2 + PY-F3).

Cada persona (P1, P2, P3) es un cliente con las tomas de su propio teléfono. El
servidor arranca de un modelo base entrenado solo con la webcam de P1, con las LSTM
congeladas: los clientes entrenan y suben únicamente la cabeza (`cabeza.py`). P2 y P3
nunca entran al backbone, que es lo que hace limpio el experimento para ellos.

Experimentos (por semilla):

A. Federado con una persona afuera (LOPO federado). Para cada persona p:
     base        el modelo del servidor, sin federar, evaluado en p
     fedavg      FedAvg de la cabeza con las otras dos personas, evaluado en p
     fedavg_8b   lo mismo subiendo deltas cuantizados a 8 bits
     techo       la cabeza entrenada con los datos de las otras dos juntos (centralizado)
     piso        la cabeza entrenada con una sola de las otras personas (promedio)
   Más la curva por ronda y las rondas hasta llegar al 95 % del techo.

B. Personalización de un usuario nuevo: el modelo federado sin p se ajusta con la
   tanda 1 de p y se evalúa con su tanda 2 (contra el mismo modelo sin ajustar).

C. Personalización de un miembro: FedAvg con la tanda 1 de las tres personas; cada
   una evalúa en su tanda 2 el modelo global y el global ajustado con su tanda 1.

B y C se comparan con un t-test pareado sobre la media de cada persona. Las semillas
no cuentan como observaciones: repiten a las mismas personas, y tratarlas como
independientes (n = personas × semillas) achica el valor p sin que haya más evidencia.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH, ver CLAUDE.md):
    python federated/run_personas.py --semillas 0 1 2
    python federated/run_personas.py --rehacer federated/resultados_personas_<corrida>
El segundo rehace el resumen de una corrida ya hecha, sin entrenar.
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
sys.path.insert(0, AQUI)
sys.path.insert(1, RAIZ)
sys.path.insert(2, os.path.join(RAIZ, "scripts"))

import agregacion as ag  # noqa: E402
import cabeza  # noqa: E402
from data import partition_subclientes  # noqa: E402


def a1h(y, c):
    import tensorflow as tf
    return tf.keras.utils.to_categorical(y, c)


def acc(modelo, X, y1h):
    return float(modelo.evaluate(X, y1h, verbose=0)[1])


def entrenar_cabeza(base, X, y1h, epocas, d, c, lr):
    m = cabeza.construir(base, d.X.shape[1], d.X.shape[2], c, lr)
    m.fit(X, y1h, epochs=epocas, batch_size=8, shuffle=True, verbose=0)
    return m


def rondas_hasta(historia, objetivo):
    for r, a in zip(historia["ronda"], historia["accuracy"], strict=True):
        if a >= objetivo:
            return r
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--rondas", type=int, default=40)
    ap.add_argument("--epocas-locales", type=int, default=2)
    ap.add_argument("--epocas-personalizacion", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--etiqueta", default="personas")
    ap.add_argument("--sub-clientes", type=int, default=1,
                    help="parte a cada persona en S clientes (data.partition_subclientes); 1 = una por persona")
    ap.add_argument("--agregacion", choices=ag.ESTRATEGIAS, default="fedavg")
    ap.add_argument("--beta", type=float, default=ag.BETA_POR_DEFECTO)
    ap.add_argument("--krum-f", type=int, default=ag.F_POR_DEFECTO)
    ap.add_argument("--krum-m", type=int, default=ag.M_POR_DEFECTO)
    ap.add_argument("--rehacer", metavar="CARPETA",
                    help="rehace resumen.md y el resumen del JSON de una corrida ya hecha, sin entrenar")
    args = ap.parse_args()
    if args.rehacer:
        rehacer(args.rehacer, ap)
        return
    parametros = {"beta": args.beta, "f": args.krum_f, "m": args.krum_m}
    if args.agregacion == "krum":
        # En A federan las otras dos personas, así que hay 2·S clientes (en C, 3·S).
        motivo = ag.validar_krum(2 * args.sub_clientes, args.krum_f, args.krum_m)
        if motivo:
            raise SystemExit(f"{motivo} en el experimento A: subí --sub-clientes")

    import tensorflow as tf
    from scipy.stats import ttest_rel
    from validar import entrenar

    from lsa import dataset as ds

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    salida = os.path.join(RAIZ, "federated", f"resultados_personas_{date.today().isoformat()}_{args.etiqueta}")
    os.makedirs(salida, exist_ok=True)
    d = ds.cargar(incluir_nada=True)
    C = len(d.clases)
    tel = d.fuente == "telefono"
    personas = sorted(set(d.persona[tel].tolist()))
    print(d.resumen(), "| clientes:", personas, flush=True)
    total_ep = args.rondas * args.epocas_locales  # mismo presupuesto de épocas para techo y piso

    filas_a, filas_b, filas_c, curvas = [], [], [], []
    for semilla in args.semillas:
        t0 = time.time()
        tf.keras.utils.set_random_seed(semilla)
        web = d.fuente == "webcam"
        base = entrenar(d.X[web], d.y[web], C, semilla, 300).get_weights()

        def datos(mascara):
            return d.X[mascara], a1h(d.y[mascara], C)

        def clientes_de(mascara):
            """Los clientes del federado sobre esas tomas: uno por persona, o S por persona."""
            idx = np.where(mascara)[0]
            partes, _ = partition_subclientes(d.persona[idx], d.y[idx], args.sub_clientes, seed=semilla)
            return [(d.X[idx[q]], a1h(d.y[idx[q]], C)) for q in partes]

        for p in personas:
            Xp, yp = datos(tel & (d.persona == p))
            otros = [q for q in personas if q != p]
            # El piso es siempre "una persona sola", aunque el federado parta a cada una.
            por_persona = [datos(tel & (d.persona == q)) for q in otros]
            clientes = clientes_de(tel & np.isin(d.persona, otros))

            m = cabeza.construir(base, d.X.shape[1], d.X.shape[2], C, args.lr)
            fila = {"semilla": semilla, "persona": p, "base": acc(m, Xp, yp), "clientes": len(clientes)}
            final, hist = cabeza.fedavg(m, clientes, Xp, yp, args.rondas, args.epocas_locales, semilla=semilla,
                                        agregacion=args.agregacion, **parametros)
            fila["fedavg"] = hist["accuracy"][-1]
            fila["bytes_fedavg"] = hist["bytes_subida"][-1]
            m8 = cabeza.construir(base, d.X.shape[1], d.X.shape[2], C, args.lr)
            _, hist8 = cabeza.fedavg(m8, clientes, Xp, yp, args.rondas, args.epocas_locales, bits=8, semilla=semilla,
                                     agregacion=args.agregacion, **parametros)
            fila["fedavg_8b"] = hist8["accuracy"][-1]
            fila["bytes_fedavg_8b"] = hist8["bytes_subida"][-1]
            Xu = np.concatenate([x for x, _ in por_persona])
            yu = np.concatenate([y for _, y in por_persona])
            fila["techo"] = acc(entrenar_cabeza(base, Xu, yu, total_ep, d, C, args.lr), Xp, yp)
            fila["piso"] = float(np.mean([acc(entrenar_cabeza(base, x, y, total_ep, d, C, args.lr), Xp, yp)
                                          for x, y in por_persona]))
            fila["rondas_95_techo"] = rondas_hasta(hist, 0.95 * fila["techo"])
            fila["n_parametros_cabeza"] = cabeza.n_parametros_cabeza(m)
            filas_a.append(fila)
            curvas.append({"semilla": semilla, "persona": p, "accuracy": hist["accuracy"],
                           "accuracy_8b": hist8["accuracy"], "bytes": hist["bytes_subida"],
                           "bytes_8b": hist8["bytes_subida"]})

            # B: usuario nuevo. El modelo federado sin p se ajusta con su tanda 1.
            t1, t2 = d.sesion == f"{p}-A1", d.sesion == f"{p}-A2"
            X1, y1 = datos(tel & t1)
            X2, y2 = datos(tel & t2)
            cabeza.poner_cabeza(m, final)
            sin = acc(m, X2, y2)
            cabeza.entrenar_local(m, final, X1, y1, args.epocas_personalizacion)
            filas_b.append({"semilla": semilla, "persona": p, "global": sin, "personalizado": acc(m, X2, y2)})
            print(f"  semilla {semilla} {p}: base {fila['base']:.3f} fedavg {fila['fedavg']:.3f} "
                  f"8b {fila['fedavg_8b']:.3f} techo {fila['techo']:.3f} piso {fila['piso']:.3f} | "
                  f"nuevo: {sin:.3f} -> {filas_b[-1]['personalizado']:.3f}", flush=True)

        # C: miembros. FedAvg con la tanda 1 de todos; cada uno evalúa en su tanda 2.
        clientes_t1 = [datos(tel & (d.sesion == f"{q}-A1")) for q in personas]
        X2_todos, y2_todos = datos(tel & np.isin(d.sesion, [f"{q}-A2" for q in personas]))
        mc = cabeza.construir(base, d.X.shape[1], d.X.shape[2], C, args.lr)
        federan_c = clientes_de(tel & np.isin(d.sesion, [f"{q}-A1" for q in personas]))
        global_c, _ = cabeza.fedavg(mc, federan_c, X2_todos, y2_todos, args.rondas, args.epocas_locales,
                                    semilla=semilla, agregacion=args.agregacion, **parametros)
        for q, (X1, y1) in zip(personas, clientes_t1, strict=True):
            X2, y2 = datos(tel & (d.sesion == f"{q}-A2"))
            cabeza.poner_cabeza(mc, global_c)
            sin = acc(mc, X2, y2)
            cabeza.entrenar_local(mc, global_c, X1, y1, args.epocas_personalizacion)
            filas_c.append({"semilla": semilla, "persona": q, "global": sin, "personalizado": acc(mc, X2, y2)})
        print(f"semilla {semilla} lista ({time.time() - t0:.0f} s)", flush=True)

        # Se guarda después de cada semilla: si la corrida se corta, lo hecho queda.
        resumen = resumir(filas_a, filas_b, filas_c, ttest_rel)
        with open(os.path.join(salida, "resultados.json"), "w", encoding="utf-8") as f:
            json.dump({"config": vars(args), "datos": d.resumen(), "resumen": resumen, "lopo": filas_a,
                       "usuario_nuevo": filas_b, "miembros": filas_c, "curvas": curvas}, f, indent=2,
                      ensure_ascii=False)
        escribir_md(salida, resumen, args)
    print(f"\n-> {salida}")


def media_desvio(v):
    v = [x for x in v if x is not None]
    return {"media": float(np.mean(v)), "desvio": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}


def resumir(a, b, c, ttest_rel):
    out = {"lopo": {k: media_desvio([f[k] for f in a]) for k in
                    ("base", "fedavg", "fedavg_8b", "techo", "piso", "rondas_95_techo")}}
    out["lopo"]["bytes_fedavg"] = int(np.mean([f["bytes_fedavg"] for f in a]))
    out["lopo"]["bytes_fedavg_8b"] = int(np.mean([f["bytes_fedavg_8b"] for f in a]))
    out["lopo"]["por_persona"] = {p: {k: media_desvio([f[k] for f in a if f["persona"] == p])["media"]
                                      for k in ("base", "fedavg", "techo", "piso")}
                                  for p in sorted({f["persona"] for f in a})}
    for nombre, filas in (("usuario_nuevo", b), ("miembros", c)):
        g = [f["global"] for f in filas]
        pz = [f["personalizado"] for f in filas]
        r = {"global": media_desvio(g), "personalizado": media_desvio(pz), "mejora_media": float(np.mean(
            np.array(pz) - np.array(g)))}
        # La prueba va sobre la media de cada persona: las semillas no son observaciones independientes.
        personas = sorted({f["persona"] for f in filas})
        gp = [float(np.mean([f["global"] for f in filas if f["persona"] == p])) for p in personas]
        zp = [float(np.mean([f["personalizado"] for f in filas if f["persona"] == p])) for p in personas]
        r["mejora_por_persona"] = {p: z - g0 for p, z, g0 in zip(personas, zp, gp, strict=True)}
        if len(personas) > 1:
            t = ttest_rel(zp, gp)
            r["t"], r["p"], r["n_prueba"] = float(t.statistic), float(t.pvalue), len(personas)
        out[nombre] = r
    return out


def rehacer(carpeta, ap):
    """Vuelve a calcular el resumen de una corrida guardada, con las filas que ya tiene."""
    from scipy.stats import ttest_rel

    ruta = os.path.join(carpeta, "resultados.json")
    with open(ruta, encoding="utf-8") as f:
        res = json.load(f)
    # Las corridas viejas no guardaron las opciones que se agregaron después: valen los valores por defecto.
    args = ap.parse_args([])
    for k, v in res["config"].items():
        setattr(args, k, v)
    res["resumen"] = resumir(res["lopo"], res["usuario_nuevo"], res["miembros"], ttest_rel)
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    escribir_md(carpeta, res["resumen"], args)
    print(f"-> {carpeta}")


def escribir_md(salida, r, args):
    pct = lambda v: f"{100 * v:.1f}".replace(".", ",")  # noqa: E731
    lo = r["lopo"]
    md = ["# Aprendizaje federado con personas reales", "",
          f"Rondas {args.rondas}, épocas locales {args.epocas_locales}, semillas {args.semillas}. "
          f"Se entrena y se sube solo la cabeza; el backbone sale de la webcam de P1 y está congelado. "
          f"Agregación: {args.agregacion}; {args.sub_clientes} cliente(s) por persona.", "",
          "## A. Federado con una persona afuera", "", "| | Accuracy (%) |", "|---|---|"]
    federan = f"{2 * args.sub_clientes} clientes" if args.sub_clientes > 1 else "2 personas"
    for k, nombre in (("base", "Modelo del servidor, sin federar"), ("piso", "Piso: una sola persona"),
                      (("fedavg", f"FedAvg ({federan})") if args.agregacion == "fedavg"
                       else ("fedavg", f"{args.agregacion} ({federan})")),
                      ("fedavg_8b", "Lo mismo, con subida a 8 bits"),
                      ("techo", "Techo: centralizado")):
        md.append(f"| {nombre} | {pct(lo[k]['media'])} ± {pct(lo[k]['desvio'])} |")
    md += ["", f"Subida total de los clientes en toda la corrida: {lo['bytes_fedavg'] / 1e6:.2f} MB en float32, "
           f"{lo['bytes_fedavg_8b'] / 1e6:.2f} MB a 8 bits. Rondas hasta el 95 % del techo: "
           f"{lo['rondas_95_techo']['media']:.1f} (n={lo['rondas_95_techo']['n']}).", ""]
    for nombre, clave in (("B. Personalización de un usuario nuevo", "usuario_nuevo"),
                          ("C. Personalización de un miembro", "miembros")):
        x = r[clave]
        md += [f"## {nombre}", "", f"Modelo global {pct(x['global']['media'])} % → personalizado con su tanda 1 "
               f"{pct(x['personalizado']['media'])} % (mejora media {pct(x['mejora_media'])} puntos"
               + (f"; t pareada sobre la media de cada persona, n = {x['n_prueba']}: t = {x['t']:.2f}, "
                  f"p = {x['p']:.4f}" if "p" in x else "") + "). Por persona: "
               + ", ".join(f"{p} {pct(v)}" for p, v in x["mejora_por_persona"].items()) + ".", ""]
    with open(os.path.join(salida, "resumen.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))


if __name__ == "__main__":
    main()
