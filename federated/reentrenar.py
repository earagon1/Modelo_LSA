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
  4. Controles antes de publicar. Técnicos (ni --forzar los saltea): sin ops Flex,
     paridad < 1e-4 y que cada seña tenga tomas en el test fijo. De calidad (--forzar
     sí): que el paquete nuevo no acierte menos que lo que hoy usan los teléfonos en el
     test fijo (sobre las señas que los dos conocen). "Lo que usan" es la mejor entre la
     cabeza del paquete vigente y la última de las rondas. Si falla, no publica: deja el
     intento en paquetes/rechazado_<N>/.

Huecos que cerró la auditoría (ML-M8): --forzar publicaba aunque hubiera ops Flex o sin
paridad; se comparaba contra la cabeza del paquete y no contra la de las rondas (que es
la que tienen los teléfonos); una sola toma aportada con una etiqueta nueva creaba una
clase para siempre; las señas nuevas no tenían tomas en el test fijo, así que nadie
medía si se aprendían; las tomas aportadas no se revalidaban.
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
from servidor import ultimo_paquete, validar_muestra  # noqa: E402

from lsa.contrato import FEATURES_VERSION  # noqa: E402

CALIBRACION_APP = os.path.join(os.path.dirname(os.path.dirname(RAIZ)), "TraductorLSA", "app", "src", "main",
                               "assets", "calibracion.json")
FRACCION_TEST = 0.15
# Una seña nueva entra al paquete recién con estas tomas aportadas: con menos, el 15 % del
# test fijo es una o dos tomas y la clase queda para siempre (las clases solo se agregan).
MIN_TOMAS_SENA_NUEVA = 20


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


def senas_nuevas_aceptadas(previas: list[str], conteo: dict[str, int],
                           minimo: int = MIN_TOMAS_SENA_NUEVA) -> tuple[list[str], dict[str, int]]:
    """Qué señas nuevas tienen tomas suficientes para entrar; las otras esperan al próximo."""
    nuevas = {c: n for c, n in conteo.items() if c not in previas}
    return sorted(c for c, n in nuevas.items() if n >= minimo), {c: n for c, n in nuevas.items() if n < minimo}


def leer_aportadas(rutas: list[str]) -> tuple[list[str], list[str]]:
    """Etiquetas de las tomas aportadas y los problemas encontrados. Se revalidan acá
    aunque el servidor ya las haya validado: el archivo puede venir de una versión vieja
    del servidor o haberse copiado a mano."""
    etiquetas, problemas = [], []
    for ruta in rutas:
        with open(ruta, encoding="utf-8") as f:
            for n, linea in enumerate(f, 1):
                if not linea.strip():
                    continue
                try:
                    m = json.loads(linea)
                except ValueError:
                    problemas.append(f"{os.path.basename(ruta)}:{n}: no es JSON")
                    continue
                motivo = validar_muestra(m, m.get("participant_id")) if isinstance(m, dict) else "no es un objeto"
                if motivo:
                    problemas.append(f"{os.path.basename(ruta)}:{n}: {motivo}")
                else:
                    etiquetas.append(m["label"])
    return etiquetas, problemas


def ampliar_test_fijo(test: set[str], uuids: np.ndarray, y: np.ndarray, clases: list[str],
                      semilla: int = 0) -> list[str]:
    """Tomas para el test fijo de las señas que no tienen ninguna (las nuevas): el 15 % de
    las suyas, al menos una. Las que ya estaban no se tocan, así el test sigue siendo el
    mismo para las señas viejas."""
    rng = np.random.default_rng(semilla)
    en_test = np.isin(uuids, list(test))
    nuevos = []
    for c in range(len(clases)):
        if np.any(en_test & (y == c)):
            continue
        idx = np.where((y == c) & (uuids != ""))[0]
        if len(idx) < 2:
            continue  # sin al menos una para entrenar y otra para medir, la atrapa el control
        k = max(1, round(FRACCION_TEST * len(idx)))
        nuevos += uuids[rng.choice(idx, k, replace=False)].tolist()
    return sorted(nuevos)


def cabeza_de_las_rondas(directorio: str, vigente: int):
    """(versión, pesos) de la última cabeza de las rondas sobre el paquete vigente (la que
    tienen los teléfonos), o None."""
    try:
        with open(os.path.join(directorio, "estado.json"), encoding="utf-8") as f:
            estado = json.load(f)
        if estado.get("paquete") != vigente or not estado.get("version"):
            return None
        with open(os.path.join(directorio, "cabezas", f"v{estado['version']}.json"), encoding="utf-8") as f:
            return estado["version"], pr.cabeza_desde_json(json.load(f))[0]
    except (OSError, ValueError, KeyError):
        return None


def evaluar_controles(sin_flex: bool, paridad: bool, senas_sin_test: list[str], acc_nueva_comun,
                      acc_referencia, tolerancia: float) -> tuple[dict, dict]:
    """(técnicos, de calidad). --forzar solo puede saltear los de calidad."""
    tecnicos = {"sin_flex": sin_flex, "paridad_backbone": paridad, "cada_sena_con_test": not senas_sin_test}
    calidad = {"no_empeora": acc_referencia is None or acc_nueva_comun >= acc_referencia - tolerancia}
    return tecnicos, calidad


def se_publica(tecnicos: dict, calidad: dict, forzar: bool) -> bool:
    return all(tecnicos.values()) and (forzar or all(calidad.values()))


def sha(ruta: str) -> str:
    with open(ruta, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="directorio de trabajo del servidor")
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--epocas", type=int, default=300)
    ap.add_argument("--tolerancia", type=float, default=0.01)
    ap.add_argument("--forzar", action="store_true",
                    help="publicar aunque empeore en el test fijo (los controles técnicos no se saltean)")
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
    etiquetas_aportadas, invalidas = leer_aportadas(aportadas)
    if invalidas:
        resto = f"\n  … y {len(invalidas) - 20} más" if len(invalidas) > 20 else ""
        raise SystemExit("Tomas aportadas inválidas (sacarlas de muestras/ antes de reentrenar):\n  "
                         + "\n  ".join(invalidas[:20]) + resto)
    conteo = {c: etiquetas_aportadas.count(c) for c in set(etiquetas_aportadas)}
    nuevas, postergadas = senas_nuevas_aceptadas(clases_vig, conteo)
    if postergadas:
        print(f"Señas nuevas con menos de {MIN_TOMAS_SENA_NUEVA} tomas, quedan para otro paquete: {postergadas}",
              flush=True)
    clases = clases_del_paquete(clases_vig, nuevas)
    d = ds.cargar(incluir_nada=True, aportadas=tuple(aportadas), clases=clases)
    print(d.resumen(), flush=True)

    ruta_test = os.path.join(args.dir, "test_fijo.json")
    if not os.path.exists(ruta_test):
        with open(ruta_test, "w", encoding="utf-8") as f:
            json.dump({"descripcion": "Tomas que ningún paquete usa para entrenar (15 % del teléfono, "
                       "estratificado por seña, semilla 0)",
                       "uuids": elegir_test_fijo(d.uuid, d.y, d.fuente == "telefono")}, f, indent=1)
    with open(ruta_test, encoding="utf-8") as f:
        guardado = json.load(f)
    test_uuids = set(guardado["uuids"])
    agregados = ampliar_test_fijo(test_uuids, d.uuid, d.y, clases)
    if agregados:
        # Se guarda una sola vez: el próximo paquete mide las señas nuevas sobre las mismas tomas.
        guardado["uuids"] = sorted(test_uuids | set(agregados))
        guardado.setdefault("ampliaciones", []).append(
            {"fecha": datetime.now().isoformat(timespec="seconds"), "uuids": agregados})
        with open(ruta_test, "w", encoding="utf-8") as f:
            json.dump(guardado, f, indent=1)
        test_uuids = set(guardado["uuids"])
        print(f"Test fijo ampliado con {len(agregados)} tomas de señas que no tenían", flush=True)
    es_test = np.isin(d.uuid, list(test_uuids))
    # Una persona excluida no entra al entrenamiento con ninguna toma (las suyas del test
    # fijo siguen en el test): para el paquete es alguien que nunca vio.
    excluida = np.isin(d.persona, args.excluir_persona)
    entrena = ~es_test & ~excluida
    # Contra lo que entrena, no contra todo: una seña con todas sus tomas en el test (o de
    # una persona excluida) no se aprende aunque "tenga tomas".
    faltan = [c for i, c in enumerate(clases) if not np.any(entrena & (d.y == i))]
    if faltan:
        raise SystemExit(f"Señas sin ninguna toma para entrenar: {faltan}")
    senas_sin_test = [c for i, c in enumerate(clases) if not np.any(es_test & (d.y == i))]
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

    # Comparación con lo que hoy usan los teléfonos, sobre las señas que conoce (las nuevas no
    # puede acertarlas): la mejor entre la cabeza del paquete y la última de las rondas. Antes
    # se comparaba solo con la del paquete, y un reentrenamiento podía publicar algo peor que
    # lo que las rondas ya habían logrado.
    acc_vig = acc_rondas = acc_nueva_comun = acc_referencia = version_rondas = None
    if vigente:
        z = np.load(os.path.join(carpeta_vigente, "test_servidor.npz"))
        comun = np.isin(z["etiquetas"], np.arange(len(clases_vig)))
        emb_v, et_v = z["embeddings"][comun], z["etiquetas"][comun]
        acc_vig = float(np.mean(pr.predecir(pesos_vig, emb_v).argmax(1) == et_v))
        rondas = cabeza_de_las_rondas(args.dir, vigente)
        if rondas:
            version_rondas, pesos_rondas = rondas
            acc_rondas = float(np.mean(pr.predecir(pesos_rondas, emb_v).argmax(1) == et_v))
        acc_referencia = max(a for a in (acc_vig, acc_rondas) if a is not None)
        en_comun = y_test < len(clases_vig)
        acc_nueva_comun = float(np.mean(pr.predecir(pesos, emb_test[en_comun]).argmax(1) == y_test[en_comun]))
    acc_por_sena_nueva = {c: float(np.mean(pr.predecir(pesos, emb_test[y_test == i]).argmax(1) == i))
                          for i, c in enumerate(clases) if i >= len(clases_vig) and np.any(y_test == i)}
    tecnicos, calidad = evaluar_controles(not flex, dif < 1e-4, senas_sin_test, acc_nueva_comun,
                                          acc_referencia, args.tolerancia)
    controles = {**tecnicos, **calidad}
    detalle = ""
    if vigente:
        detalle = (f", sobre las señas viejas {acc_nueva_comun:.3f} contra {acc_referencia:.3f} "
                   f"(paquete v{vigente}: {acc_vig:.3f}"
                   + (f", cabeza de rondas v{version_rondas}: {acc_rondas:.3f}" if acc_rondas is not None else "")
                   + ")")
    if acc_por_sena_nueva:
        detalle += f" | señas nuevas: {acc_por_sena_nueva}"
    print(f"controles {controles} | accuracy test fijo: nueva {acc_nueva:.3f}{detalle}", flush=True)

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
                  # La app la compara con la suya antes de traducir (auditoría, X3).
                  "features_version": FEATURES_VERSION,
                  # Las tomas del test fijo salen de los teléfonos del piloto: las apps y los
                  # clientes emulados no las usan para entrenar ni las aportan.
                  "uuids_test": sorted(test_uuids),
                  "metricas": {"accuracy_test_fijo": acc_nueva, "accuracy_senas_previas": acc_nueva_comun,
                               "accuracy_paquete_anterior": acc_vig, "accuracy_cabeza_rondas": acc_rondas,
                               "version_cabeza_rondas": version_rondas, "accuracy_referencia": acc_referencia,
                               "accuracy_por_sena_nueva": acc_por_sena_nueva, "n_test": int(es_test.sum()),
                               "n_entrenamiento": int(entrena.sum()),
                               "tomas_aportadas": int((d.fuente == "aporte").sum()),
                               "paridad_backbone_dif_max": dif},
                  "controles": controles, "forzado": bool(args.forzar and not all(calidad.values())),
                  "senas_postergadas": postergadas, "datos": d.resumen(), "semilla": args.semilla,
                  "personas_excluidas": args.excluir_persona}
    if not se_publica(tecnicos, calidad, args.forzar):
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
