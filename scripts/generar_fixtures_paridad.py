"""Genera contract/fixtures/paridad_features_v2.json (J-02) y actualiza SHA256SUMS.

Los casos son sintéticos y deterministas (semilla fija): no hacen falta datos reales
para fijar el contrato, y así el fixture se puede publicar en los dos repos.

Uso, desde la raíz de modelo_LSA:
    python scripts/generar_fixtures_paridad.py
Después hay que copiar `contract/` entero a TraductorLSA/contract/.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa import contrato
from lsa.contrato import Mano

RAIZ_CONTRATO = contrato.RUTA_CONTRATO
SALIDA = os.path.join(RAIZ_CONTRATO, "fixtures", "paridad_features_v2.json")
rng = np.random.default_rng(20261005)

ORIENTACIONES = {"vertical": (480, 640), "horizontal": (640, 480)}


def mano_sintetica(centro_x: float) -> np.ndarray:
    """21 landmarks con forma vaga de mano alrededor de un centro, redondeados a 6
    decimales para que el JSON de entrada sea exacto en los dos lenguajes."""
    base = np.stack([
        centro_x + rng.uniform(-0.12, 0.12, 21),
        0.55 + rng.uniform(-0.2, 0.2, 21),
        rng.uniform(-0.08, 0.02, 21),
    ], axis=1)
    base[0, 2] = 0.0  # la muñeca tiene z = 0 en MediaPipe
    return np.round(base, 6)


def frame(manos_spec):
    return [Mano(etq, score, mano_sintetica(cx)) for etq, score, cx in manos_spec]


def a_json_frame(manos):
    return {"manos": [{"etiqueta": m.etiqueta, "score": m.score,
                       "landmarks": m.landmarks.tolist()} for m in manos]}


def caso(nombre, frames, ancho, alto, camara, orientacion, nota=""):
    esperado = contrato.secuencia_desde_frames(frames, ancho, alto)
    vectores = [contrato.frame_a_vector(f, ancho, alto) for f in frames]
    return {
        "nombre": nombre,
        "nota": nota,
        "entrada": {"ancho": ancho, "alto": alto, "camara": camara, "orientacion": orientacion,
                    "frames": [a_json_frame(f) for f in frames]},
        "esperado": {
            "n_validos": int(sum(bool((v != 0).any()) for v in vectores)),
            "vectores_por_frame": [v.tolist() for v in vectores],
            "secuencia": esperado.tolist(),
        },
    }


casos = []
for n in (8, 10, 15, 20, 27):
    for orient, (w, h) in ORIENTACIONES.items():
        for camara in ("frontal", "trasera"):
            # Una mano derecha que señala, con la izquierda apareciendo a veces.
            frames = []
            for i in range(n):
                spec = [("Right", 0.97, 0.40)]
                if i % 3 == 0:
                    spec.append(("Left", 0.91, 0.70))
                frames.append(frame(spec))
            casos.append(caso(f"n{n}_{orient}_{camara}", frames, w, h, camara, orient))

casos.append(caso(
    "etiqueta_repetida", [frame([("Right", 0.80, 0.35), ("Right", 0.95, 0.65)]) for _ in range(12)],
    480, 640, "frontal", "vertical",
    "Las dos manos dicen Right: la de mayor score (la segunda) se queda en rh y la otra va a lh."))
casos.append(caso(
    "etiqueta_desconocida", [frame([("Unknown", 0.90, 0.5)]) for _ in range(9)],
    480, 640, "frontal", "vertical", "Una etiqueta que no es Left/Right va al slot por defecto (rh)."))
casos.append(caso(
    "frames_vacios_intercalados",
    [frame([] if i in (0, 1, 5, 11, 12) else [("Right", 0.9, 0.45)]) for i in range(13)],
    480, 640, "frontal", "vertical",
    "Los frames sin mano se descartan antes de remuestrear: quedan 8 válidos."))
casos.append(caso(
    "un_solo_frame", [frame([("Left", 0.9, 0.5)])], 480, 640, "trasera", "vertical",
    "Con n=1 se repite el frame T veces."))
casos.append(caso(
    "cuadrada", [frame([("Left", 0.9, 0.3), ("Right", 0.9, 0.7)]) for _ in range(15)],
    720, 720, "frontal", "vertical", "Con W=H la corrección de aspecto es la identidad."))

fixture = {
    "features_version": contrato.FEATURES_VERSION,
    "T": contrato.T,
    "D": contrato.D,
    "tolerancia": contrato.CONTRATO["tolerancia_paridad"],
    "generado_por": "modelo_LSA/scripts/generar_fixtures_paridad.py",
    "casos": casos,
}
os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
with open(SALIDA, "w", encoding="utf-8", newline="\n") as f:
    json.dump(fixture, f, ensure_ascii=False, separators=(",", ":"))
    f.write("\n")

# SHA256SUMS fija el contenido de contract/ para el test de espejo de los dos repos.
lineas = []
for raiz, _, archivos in os.walk(RAIZ_CONTRATO):
    for a in sorted(archivos):
        if a == "SHA256SUMS":
            continue
        p = os.path.join(raiz, a)
        rel = os.path.relpath(p, RAIZ_CONTRATO).replace(os.sep, "/")
        with open(p, "rb") as fh:
            lineas.append(f"{hashlib.sha256(fh.read()).hexdigest()}  {rel}")
with open(os.path.join(RAIZ_CONTRATO, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(sorted(lineas, key=lambda s: s.split("  ")[1])) + "\n")

print(f"{len(casos)} casos -> {SALIDA} ({os.path.getsize(SALIDA) / 1024:.0f} KB)")
