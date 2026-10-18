"""Cliente emulado del federado por red local (PY-F7).

Hace lo mismo que la app, con las tomas de una persona de la sesión de captura:
  - baja el paquete vigente si no lo tiene (GET /paquete + archivos) y verifica los sha256;
  - en cada ronda: baja la cabeza global (GET /modelo), la entrena unas épocas con sus
    tomas (Adam, dropout de entrada, como CabezaDensa.kt), y sube el delta a 8 bits
    (POST /update) con sus métricas locales;
  - con --aportar, sube sus tomas (POST /muestras) para el reentrenamiento completo.

Sirve para probar el ciclo entero sin teléfonos y para el piloto híbrido: teléfonos
reales más clientes emulados con los datos de las otras personas (PLAN_MAESTRO §6.2).

Uso, desde la raíz de modelo_LSA, con el servidor corriendo:
    python federated/cliente_emulado.py --servidor http://127.0.0.1:8765 --persona P2 --rondas 3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.append(AQUI)
import protocolo as pr  # noqa: E402

DROPOUT = 0.5
LR = 1e-3


def pedir(url: str, cuerpo: dict | None = None, timeout: float = 60):
    datos = json.dumps(cuerpo).encode() if cuerpo is not None else None
    req = urllib.request.Request(url, datos, {"Content-Type": "application/json"} if datos else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def bajar_paquete(servidor: str, cache: str) -> tuple[dict, str]:
    """El paquete vigente en `cache/v<N>/`, bajándolo si hace falta. Verifica cada sha."""
    codigo, cuerpo = pedir(servidor + "/paquete")
    if codigo != 200:
        raise SystemExit(f"el servidor no tiene paquete ({codigo})")
    man = json.loads(cuerpo)
    carpeta = os.path.join(cache, f"v{man['paquete']}")
    os.makedirs(carpeta, exist_ok=True)
    for a in man["archivos"]:
        ruta = os.path.join(carpeta, a["nombre"])
        if os.path.exists(ruta) and hashlib.sha256(open(ruta, "rb").read()).hexdigest() == a["sha256"]:
            continue
        codigo, datos = pedir(f"{servidor}/paquete/v{man['paquete']}/{a['nombre']}")
        if codigo != 200 or hashlib.sha256(datos).hexdigest() != a["sha256"]:
            raise SystemExit(f"{a['nombre']}: descarga fallida o sha distinto")
        with open(ruta, "wb") as f:
            f.write(datos)
    return man, carpeta


def entrenar_cabeza(pesos, emb, y, n_clases, epocas, semilla):
    """Ajuste local de la cabeza: Dropout(0.5) → 64 relu → 64 relu → C softmax, L2 0,001."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(semilla)
    entrada = tf.keras.Input((emb.shape[1],))
    x = tf.keras.layers.Dropout(DROPOUT)(entrada)
    l2 = tf.keras.regularizers.l2(0.001)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=l2)(x)
    salida = tf.keras.layers.Dense(n_clases, activation="softmax")(x)
    m = tf.keras.Model(entrada, salida)
    m.set_weights(pesos)
    m.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="categorical_crossentropy")
    m.fit(emb, tf.keras.utils.to_categorical(y, n_clases), epochs=epocas, batch_size=8, verbose=0)
    return [w.astype(np.float32) for w in m.get_weights()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--servidor", required=True)
    ap.add_argument("--persona", required=True, help="P1, P2 o P3 (sus tomas de la sesión 1)")
    ap.add_argument("--rondas", type=int, default=1)
    ap.add_argument("--epocas", type=int, default=2)
    ap.add_argument("--aportar", action="store_true", help="subir también sus tomas (POST /muestras)")
    ap.add_argument("--cache", default=os.path.join(AQUI, "piloto_clientes"))
    ap.add_argument("--espera", type=float, default=2.0, help="segundos entre consultas mientras espera la ronda")
    args = ap.parse_args()

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")  # varios clientes a la vez: la GPU no hace falta
    from lsa import dataset as ds
    from lsa import dataset_v2
    from lsa.export import inferir_tflite

    servidor = args.servidor.rstrip("/")
    pid = f"emulado_{args.persona.lower()}"
    ruta_v2 = os.path.join(RAIZ, "data", "captura_E3", args.persona, "lsa_samples_v2.jsonl")
    if args.aportar:
        man, _ = bajar_paquete(servidor, os.path.join(args.cache, pid))
        excluir = set(man.get("uuids_test", []))
        muestras = [{**m, "participant_id": pid} for m in dataset_v2.leer(ruta_v2)
                    if (m.get("sesion") or {}).get("bloque") in ("A", "N") and m["uuid"] not in excluir]
        codigo, r = pedir(servidor + "/muestras", {"protocolo": pr.PROTOCOLO, "participant_id": pid,
                                                   "muestras": muestras}, timeout=300)
        print(f"{pid}: aporte de {len(muestras)} tomas -> {codigo} {json.loads(r)}", flush=True)

    ronda_hecha = -1
    for _ in range(args.rondas):
        man, carpeta = bajar_paquete(servidor, os.path.join(args.cache, pid))
        with open(os.path.join(carpeta, "backbone.tflite"), "rb") as f:
            tfl = f.read()
        d = ds.cargar(incluir_nada=True, clases=man["clases"])
        mia = (d.persona == args.persona) & (d.fuente == "telefono") & ~np.isin(d.uuid, man.get("uuids_test", []))
        emb = inferir_tflite(tfl, d.X[mia]).astype(np.float32)
        y = d.y[mia]
        # Espera a que el servidor abra una ronda nueva (la anterior la cerró el quórum).
        while True:
            m = json.loads(pedir(servidor + "/modelo")[1])
            if m["ronda"] > ronda_hecha:
                break
            time.sleep(args.espera)
        if m["backbone_sha256"] != hashlib.sha256(tfl).hexdigest():
            continue  # salió un paquete nuevo mientras tanto: se baja en la vuelta siguiente
        globales = pr.decodificar_pesos(m["pesos"])
        if pr.sha_pesos(globales) != m["sha256"]:
            raise SystemExit("la cabeza llegó corrupta")
        antes = float(np.mean(pr.predecir(globales, emb).argmax(1) == y))
        t0 = time.time()
        locales = entrenar_cabeza(globales, emb, y, len(m["clases"]), args.epocas, semilla=m["ronda"])
        despues = float(np.mean(pr.predecir(locales, emb).argmax(1) == y))
        delta = [a - b for a, b in zip(locales, globales, strict=True)]
        u = {"protocolo": pr.PROTOCOLO, "ronda": m["ronda"], "participant_id": pid, "n_muestras": len(y),
             "bits": pr.BITS, "tensores": pr.codificar_delta(delta, np.random.default_rng(m["ronda"])),
             "metricas_locales": {"accuracy_antes": antes, "accuracy_despues": despues,
                                  "segundos_entrenamiento": round(time.time() - t0, 2), "emulado": True}}
        codigo, r = pedir(servidor + "/update", u)
        print(f"{pid} ronda {m['ronda']} (paquete v{m['paquete']}): local {antes:.3f} -> {despues:.3f}, "
              f"update -> {codigo} {json.loads(r)}", flush=True)
        ronda_hecha = m["ronda"]


if __name__ == "__main__":
    main()
