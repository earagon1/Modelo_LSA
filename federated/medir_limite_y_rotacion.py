"""Mide el límite a la magnitud de los updates y la rotación antes de cuantizar.

Las dos medidas vienen de la propuesta: "limitar la magnitud de actualizaciones" (tabla
de riesgos) y las actualizaciones esbozadas de Konečný et al. (rotación aleatoria +
cuantización). Con updates honestos de verdad (la cabeza del paquete del piloto
entrenada 2 épocas con las tomas de P1, P2 y P3, como hace el teléfono en una ronda):

  1. Norma de los updates honestos: de ahí sale el límite (un múltiplo de la mediana).
  2. Error de reconstrucción de la cuantización a 8 bits, con y sin rotación, y bytes.
  3. Un cliente malicioso: accuracy del modelo agregado sin límite y con límite, sobre
     el test fijo + la sesión 2 (tomas que no participan del federado), y la distancia
     de la cabeza agregada a la que sale solo con los honestos. Dos ataques: un update
     enorme al azar (×50 la norma honesta) y uno que invierte el aprendizaje (−10 veces
     el update honesto medio).

Escribe resultados_federado_limite_rotacion/<fecha>.json y un resumen en consola.

Uso, desde la raíz de modelo_LSA:
    python federated/medir_limite_y_rotacion.py --dir federated/piloto
"""
import argparse
import json
import os
import sys
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import protocolo as pr  # noqa: E402
from cliente_emulado import entrenar_cabeza  # noqa: E402
from servidor import leer_paquete, ultimo_paquete  # noqa: E402

MULTIPLO_LIMITE = 3.0


def error_relativo(a, b):
    return pr.norma([x - y for x, y in zip(a, b, strict=True)]) / max(pr.norma(a), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--repeticiones", type=int, default=20, help="sorteos de redondeo y rotación por cliente")
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from lsa import dataset as ds
    from lsa.export import inferir_tflite

    n = ultimo_paquete(args.dir)
    man, pesos0, clases, test = leer_paquete(args.dir, n)
    with open(os.path.join(args.dir, "paquetes", f"v{n}", "backbone.tflite"), "rb") as f:
        tfl = f.read()
    d = ds.cargar(incluir_nada=True, clases=clases)
    excluir = set(man.get("uuids_test", []))

    deltas, filas = {}, []
    for p in ("P1", "P2", "P3"):
        m = (d.persona == p) & (d.fuente == "telefono") & ~np.isin(d.uuid, list(excluir))
        emb = inferir_tflite(tfl, d.X[m]).astype(np.float32)
        locales = entrenar_cabeza(pesos0, emb, d.y[m], len(clases), epocas=2, semilla=0)
        delta = [a - b for a, b in zip(locales, pesos0, strict=True)]
        deltas[p] = (delta, int(m.sum()))
        rng = np.random.default_rng(0)
        e_sin, e_con = [], []
        for k in range(args.repeticiones):
            e_sin.append(error_relativo(delta, pr.decodificar_delta(pr.codificar_delta(delta, rng))))
            e_con.append(error_relativo(delta, pr.decodificar_delta(pr.codificar_delta(delta, rng, semilla=1000 * k))))
        # La rotación ayuda cuando unos pocos valores son mucho más grandes que el resto
        # (el máximo fija la escala de los 8 bits): se mide con max|x| / rms.
        pico = lambda v: float(np.max(np.abs(v)) / np.sqrt(np.mean(np.square(v))))  # noqa: E731
        picos_sin = [pico(x.ravel()) for x in delta]
        picos_con = [pico(pr.rotar(x, 3 + i)) for i, x in enumerate(delta)]
        sin_t = pr.codificar_delta(delta, rng)
        con_t = pr.codificar_delta(delta, rng, semilla=7)
        filas.append({"persona": p, "muestras": int(m.sum()), "norma": pr.norma(delta),
                      "error_sin_rotacion": float(np.mean(e_sin)), "error_con_rotacion": float(np.mean(e_con)),
                      "bytes_sin_rotacion": pr.bytes_payload(sin_t), "bytes_con_rotacion": pr.bytes_payload(con_t),
                      "bytes_float32": int(sum(x.size for x in delta) * 4),
                      "pico_max_sobre_rms_sin_rotacion": picos_sin, "pico_max_sobre_rms_con_rotacion": picos_con})
        f = filas[-1]
        print(f"{p}: norma {f['norma']:.3f} | error 8 bits sin rotación {100 * f['error_sin_rotacion']:.2f} %, "
              f"con rotación {100 * f['error_con_rotacion']:.2f} % | bytes {f['bytes_sin_rotacion']} -> "
              f"{f['bytes_con_rotacion']} (float32 {f['bytes_float32']}) | max/rms por tensor "
              f"{[round(v, 1) for v in picos_sin]} -> {[round(v, 1) for v in picos_con]}", flush=True)

    normas = [f["norma"] for f in filas]
    limite = MULTIPLO_LIMITE * float(np.median(normas))
    print(f"límite = {MULTIPLO_LIMITE} × mediana de las normas honestas = {limite:.3f}", flush=True)

    # Evaluación: test fijo + sesión 2 (ninguna de las dos entra al federado).
    emb_t, y_t = test
    b = ds.cargar(incluir_nada=True, bloques=("B",), clases=clases)
    emb_e = np.concatenate([emb_t, inferir_tflite(tfl, b.X).astype(np.float32)])
    y_e = np.concatenate([y_t, b.y])
    acc = lambda w: float(np.mean(pr.predecir(w, emb_e).argmax(1) == y_e))  # noqa: E731
    honestos = list(deltas.values())
    referencia = pr.agregar(pesos0, honestos)
    distancia = lambda w: pr.norma([a - c for a, c in zip(w, referencia, strict=True)])  # noqa: E731
    rng = np.random.default_rng(1)
    ruido = [rng.normal(0, 1, x.shape).astype(np.float32) for x in pesos0]
    medio = [np.mean([dl[i] for dl, _ in honestos], axis=0) for i in range(len(pesos0))]
    ataques = {
        "al_azar_x50": [x * np.float32(50 * float(np.median(normas)) / pr.norma(ruido)) for x in ruido],
        "invertido_x10": [-10 * x for x in medio],
    }
    resultados_ataque = {"evaluacion_tomas": len(y_e), "antes": acc(pesos0), "solo_honestos": acc(referencia),
                         "solo_honestos_con_limite": acc(pr.agregar(pesos0, [(pr.recortar(dl, limite)[0], k)
                                                                             for dl, k in honestos]))}
    print(f"evaluación sobre {len(y_e)} tomas (test fijo + sesión 2): antes {100 * resultados_ataque['antes']:.1f} %, "
          f"solo honestos {100 * resultados_ataque['solo_honestos']:.1f} %, honestos con límite "
          f"{100 * resultados_ataque['solo_honestos_con_limite']:.1f} %", flush=True)
    for nombre, delta_mal in ataques.items():
        todos = [*honestos, (delta_mal, deltas["P2"][1])]
        sin = pr.agregar(pesos0, todos)
        con = pr.agregar(pesos0, [(pr.recortar(dl, limite)[0], k) for dl, k in todos])
        resultados_ataque[nombre] = {"sin_limite": acc(sin), "con_limite": acc(con),
                                     "distancia_sin_limite": distancia(sin), "distancia_con_limite": distancia(con)}
        r = resultados_ataque[nombre]
        print(f"ataque {nombre}: sin límite {100 * r['sin_limite']:.1f} % (se aleja {r['distancia_sin_limite']:.2f}), "
              f"con límite {100 * r['con_limite']:.1f} % (se aleja {r['distancia_con_limite']:.2f})", flush=True)

    salida = os.path.join(RAIZ, "resultados_federado_limite_rotacion")
    os.makedirs(salida, exist_ok=True)
    with open(os.path.join(salida, f"{date.today().isoformat()}.json"), "w", encoding="utf-8") as f:
        json.dump({"paquete": n, "clientes": filas, "multiplo_limite": MULTIPLO_LIMITE, "limite": limite,
                   "ataques": resultados_ataque,
                   "repeticiones": args.repeticiones}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
