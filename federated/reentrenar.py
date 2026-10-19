"""Reentrena el modelo completo en el servidor y publica un paquete para las apps.

Es la segunda vía del federado por red local (plan/federado_completo.md). El servidor
(`federated/servidor.py`) junta las tomas que las apps aportan con consentimiento
(muestras/*.jsonl, solo landmarks). Este script:

  1. Arma el entrenamiento con todas las fuentes de siempre (webcam + sesión de
     captura) más las tomas aportadas, sin repetir tomas (por uuid). Si las tomas
     aportadas traen una seña nueva, el paquete sale con una clase más.
  2. Deja afuera un test fijo (test_fijo.json, elegido una sola vez: 15 % de las tomas
     del teléfono, estratificado por seña). Ningún paquete entrena con él, así que
     todos se comparan sobre las mismas tomas.
  3. Entrena, parte el modelo en backbone (las LSTM, TFLite sin Flex) y cabeza (las
     Dense, JSON para CabezaDensa.kt) y verifica la paridad del backbone con Keras.
  4. Controles antes de publicar: sin ops Flex, paridad < 1e-4 y que el paquete nuevo
     no acierte menos que el vigente en el test fijo (sobre las señas que los dos
     conocen). Si falla, no publica: deja el intento en paquetes/rechazado_<N>/.
  5. Publica paquetes/v<N>/: backbone.tflite, cabeza.json, calibracion.json,
     test_servidor.npz y manifiesto.json (con el sha256 y los bytes de cada archivo).
     Se escribe en una carpeta temporal y se renombra al final: el servidor nunca ve
     un paquete a medias.

La calibración (temperatura) se hereda del paquete anterior o de la app: recalcularla
pide la validación por persona (scripts/calibrar.py) y no entra en cada reentrenamiento.
El manifiesto lo declara.

Uso, desde la raíz de modelo_LSA (con la GPU en el PATH):
    python federated/reentrenar.py --dir federated/piloto
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(1, os.path.join(RAIZ, "scripts"))
sys.path.append(AQUI)

import protocolo as pr  # noqa: E402
from servidor import ultimo_paquete  # noqa: E402

CALIBRACION_APP = os.path.join(os.path.dirname(os.path.dirname(RAIZ)), "TraductorLSA", "app", "src", "main",
                               "assets", "calibracion.json")
FRACCION_TEST = 0.15


def elegir_test_fijo(uuids: np.ndarray, y: np.ndarray, es_telefono: np.ndarray, semilla: int = 0) -> list[str]:
    """15 % de las tomas del teléfono, estratificado por seña. Se elige una vez y se guarda."""
    rng = np.random.default_rng(semilla)
    elegidos = []
    for c in np.unique(y[es_telefono]):
        idx = np.where(es_telefono & (y == c) & (uuids != ""))[0]
        k = max(1, round(FRACCION_TEST * len(idx)))
        elegidos += uuids[rng.choice(idx, k, replace=False)].tolist()
    return sorted(elegidos)


def clases_del_paquete(previas: list[str], aportadas: list[str]) -> list[str]:
    """Las clases del paquete vigente en su orden (el orden es el índice de salida) y las
    señas nuevas al final, en orden alfabético para que sea reproducible."""
    return previas + sorted({c for c in aportadas if c not in previas})


def sha(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directorio de trabajo del servidor")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--tolerancia", type=float, default=0.01)
    ap.add_argument("--forzar", action="store_true", help="publicar aunque no supere el control de calidad")
    ap.add_argument("--excluir-persona", nargs="*", default=[],
                    help="personas que el paquete no ve (experimento: alguien nuevo que se suma por el federado)")
    args = ap.parse_args()

    import tensorflow as tf
    from exportar_cabeza import a_json_cabeza, separar
    from validar import entrenar

    from lsa import dataset as ds
    from lsa.export import convertir, inferir_tflite, ops_flex

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)

    os.makedirs(os.path.join(args.dir, "paquetes"), exist_ok=True)
    vigente = ultimo_paquete(args.dir)
    carpeta_vigente = os.path.join(args.dir, "paquetes", f"v{vigente}") if vigente else None
    if vigente:
        with open(os.path.join(carpeta_vigente, "cabeza.json"), encoding="utf-8") as f:
            pesos_vig, clases_vig = pr.cabeza_desde_json(json.load(f))
    else:
        pesos_vig, clases_vig = None, ds.clases_del_modelo(incluir_nada=True)

    aportadas = sorted(glob.glob(os.path.join(args.dir, "muestras", "*.jsonl")))
    etiquetas_aportadas = []
    for ruta in aportadas:
        with open(ruta, encoding="utf-8") as f:
            etiquetas_aportadas += [json.loads(linea)["label"] for linea in f if linea.strip()]
    clases = clases_del_paquete(clases_vig, etiquetas_aportadas)
    d = ds.cargar(incluir_nada=True, aportadas=tuple(aportadas), clases=clases)
    print(d.resumen(), flush=True)
    faltan = [c for c in clases if not np.any(d.y == clases.index(c))]
    if faltan:
        raise SystemExit(f"Señas sin ninguna toma: {faltan}")

    ruta_test = os.path.join(args.dir, "test_fijo.json")
    if not os.path.exists(ruta_test):
        with open(ruta_test, "w", encoding="utf-8") as f:
            json.dump({"descripcion": "Tomas que ningún paquete usa para entrenar (15 % del teléfono, "
                       "estratificado por seña, semilla 0)",
                       "uuids": elegir_test_fijo(d.uuid, d.y, d.fuente == "telefono")}, f, indent=1)
    with open(ruta_test, encoding="utf-8") as f:
        test_uuids = set(json.load(f)["uuids"])
    es_test = np.isin(d.uuid, list(test_uuids))
    # Una persona excluida no entra al entrenamiento con ninguna toma (las suyas del test
    # fijo siguen en el test): para el paquete es alguien que nunca vio.
    excluida = np.isin(d.persona, args.excluir_persona)
    entrena = ~es_test & ~excluida
    print(f"entrenamiento {int(entrena.sum())}, test fijo {int(es_test.sum())}, "
          f"excluidas {args.excluir_persona or 'ninguna'} ({int((excluida & ~es_test).sum())} tomas)", flush=True)

    modelo = entrenar(d.X[entrena], d.y[entrena], len(clases), args.semilla, args.epocas)
    backbone, densas = separar(modelo)
    tfl = convertir(backbone, 15, 126, "f32")
    flex = ops_flex(tfl)
    emb_test = inferir_tflite(tfl, d.X[es_test]).astype(np.float32)
    dif = float(np.abs(backbone.predict(d.X[es_test][:16], verbose=0) - emb_test[:16]).max())
    cabeza = a_json_cabeza(densas, clases)
    pesos, _ = pr.cabeza_desde_json(cabeza)
    y_test = d.y[es_test]
    acc_nueva = float(np.mean(pr.predecir(pesos, emb_test).argmax(1) == y_test))

    # Comparación con el paquete vigente, sobre las señas que conoce (las nuevas no puede acertarlas).
    acc_vig = acc_nueva_comun = None
    if vigente:
        z = np.load(os.path.join(carpeta_vigente, "test_servidor.npz"))
        comun = np.isin(z["etiquetas"], np.arange(len(clases_vig)))
        acc_vig = float(np.mean(pr.predecir(pesos_vig, z["embeddings"][comun]).argmax(1) == z["etiquetas"][comun]))
        en_comun = y_test < len(clases_vig)
        acc_nueva_comun = float(np.mean(pr.predecir(pesos, emb_test[en_comun]).argmax(1) == y_test[en_comun]))
    controles = {"sin_flex": not flex, "paridad_backbone": dif < 1e-4,
                 "no_empeora": acc_vig is None or acc_nueva_comun >= acc_vig - args.tolerancia}
    print(f"controles {controles} | accuracy test fijo: nueva {acc_nueva:.3f}"
          + (f", sobre las señas viejas {acc_nueva_comun:.3f} contra {acc_vig:.3f} del v{vigente}" if vigente else ""),
          flush=True)

    n = vigente + 1
    temporal = os.path.join(args.dir, "paquetes", f".tmp_v{n}")
    shutil.rmtree(temporal, ignore_errors=True)
    os.makedirs(temporal)
    with open(os.path.join(temporal, "backbone.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(temporal, "cabeza.json"), "w", encoding="utf-8") as f:
        json.dump(cabeza, f)
    np.savez(os.path.join(temporal, "test_servidor.npz"), embeddings=emb_test, etiquetas=y_test)
    origen_cal = os.path.join(carpeta_vigente, "calibracion.json") if vigente else CALIBRACION_APP
    with open(origen_cal, encoding="utf-8") as f:
        cal = json.load(f)
    calibracion = {"temperatura": cal["temperatura"], "umbral": cal["umbral"],
                   "modelo_sha256": hashlib.sha256(tfl).hexdigest(),
                   "origen": f"heredada de {os.path.relpath(origen_cal, RAIZ)} (no se recalcula en cada paquete)"}
    with open(os.path.join(temporal, "calibracion.json"), "w", encoding="utf-8") as f:
        json.dump(calibracion, f, indent=1, ensure_ascii=False)

    archivos = [{"nombre": a, "sha256": sha(os.path.join(temporal, a)),
                 "bytes": os.path.getsize(os.path.join(temporal, a))}
                for a in ("backbone.tflite", "cabeza.json", "calibracion.json")]
    manifiesto = {"protocolo": pr.PROTOCOLO, "paquete": n, "creado": datetime.now().isoformat(timespec="seconds"),
                  "clases": clases, "senas_nuevas": clases[len(clases_vig):], "archivos": archivos,
                  "backbone_sha256": hashlib.sha256(tfl).hexdigest(), "cabeza_sha256": pr.sha_pesos(pesos),
                  "entrada": {"T": 15, "D": 126, "features": "contrato v2"},
                  # Las tomas del test fijo salen de los teléfonos del piloto: las apps y los
                  # clientes emulados no las usan para entrenar ni las aportan.
                  "uuids_test": sorted(test_uuids),
                  "metricas": {"accuracy_test_fijo": acc_nueva, "accuracy_senas_previas": acc_nueva_comun,
                               "accuracy_paquete_anterior": acc_vig, "n_test": int(es_test.sum()),
                               "n_entrenamiento": int(entrena.sum()),
                               "tomas_aportadas": int((d.fuente == "aporte").sum()),
                               "paridad_backbone_dif_max": dif},
                  "controles": controles, "datos": d.resumen(), "semilla": args.semilla,
                  "personas_excluidas": args.excluir_persona}
    if not all(controles.values()) and not args.forzar:
        destino = os.path.join(args.dir, "paquetes", f"rechazado_{n}_{datetime.now():%Y%m%d_%H%M%S}")
        with open(os.path.join(temporal, "rechazo.json"), "w", encoding="utf-8") as f:
            json.dump(manifiesto, f, indent=1, ensure_ascii=False)
        os.rename(temporal, destino)
        print(f"NO se publica: {controles}. Intento en {destino}", flush=True)
        return 1
    with open(os.path.join(temporal, "manifiesto.json"), "w", encoding="utf-8") as f:
        json.dump(manifiesto, f, indent=1, ensure_ascii=False)
    os.rename(temporal, os.path.join(args.dir, "paquetes", f"v{n}"))
    print(f"Paquete v{n} publicado ({len(clases)} clases, backbone {len(tfl) / 1024:.0f} KB). "
          "El servidor lo reparte solo.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
