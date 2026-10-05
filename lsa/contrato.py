"""Implementación de referencia del contrato de features v2 (`contract/features_v2.md`).

Es la única definición en Python de cómo se pasa de landmarks crudos a la entrada
15×126 del modelo. La app Android implementa lo mismo en Kotlin (tarea T-A11), y los
dos lados se comparan contra `contract/fixtures/paridad_features_v2.json`.

Las cuentas imitan a propósito el orden y la precisión de la app (float32, mismos
pasos intermedios): la paridad se exige con una tolerancia de 1e-6, así que un
"equivalente matemático" calculado en float64 no alcanza.

No importa TensorFlow, MediaPipe ni OpenCV: se usa en tests sin GPU ni cámara.
"""
from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

RUTA_CONTRATO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "contract")

with open(os.path.join(RUTA_CONTRATO, "features_v2.json"), encoding="utf-8") as _f:
    CONTRATO = json.load(_f)

FEATURES_VERSION: str = CONTRATO["features_version"]
T: int = CONTRATO["secuencia"]["T"]
D: int = CONTRATO["vector"]["dim"]
LANDMARKS = CONTRATO["vector"]["landmarks_por_mano"]
MIN_FRAMES_VALIDOS: int = CONTRATO["secuencia"]["min_frames_validos"]


@dataclass(frozen=True)
class Mano:
    """Una mano tal como la devuelve HandLandmarker: la etiqueta de mayor score,
    ese score y los 21 landmarks normalizados (x, y, z) de la imagen sin espejar."""
    etiqueta: str
    score: float
    landmarks: np.ndarray  # (21, 3)


def corregir_aspecto(landmarks: np.ndarray, ancho: int, alto: int) -> np.ndarray:
    """Lleva x,y al lienzo cuadrado virtual (regla 4). z no cambia.

    Mismo orden de operaciones que `FeatureBuilder.toVector126` de la app: la escala
    y el offset se calculan en float32 y después se aplica `x·s + o`.
    """
    lm = np.asarray(landmarks, dtype=np.float32)
    m = np.float32(max(ancho, alto))
    sx, sy = np.float32(ancho) / m, np.float32(alto) / m
    ox, oy = (np.float32(1) - sx) / np.float32(2), (np.float32(1) - sy) / np.float32(2)
    out = lm.copy()
    out[:, 0] = lm[:, 0] * sx + ox
    out[:, 1] = lm[:, 1] * sy + oy
    return out


def asignar_lateralidad(manos: Sequence[Mano]) -> dict[str, Mano]:
    """Decide a qué slot va cada mano (regla 3).

    Con dos manos de la misma etiqueta, la de mayor score se queda con su slot y la
    otra va al libre. Sin esta regla, la segunda pisaría a la primera y se perdería
    una mano entera del vector.
    """
    tabla = CONTRATO["lateralidad"]["etiqueta_a_slot"]
    defecto = CONTRATO["lateralidad"]["slot_por_defecto"]
    candidatas = list(manos)[:2]
    # Orden estable: con scores iguales, gana la que vino primero.
    orden = sorted(range(len(candidatas)), key=lambda i: -candidatas[i].score)
    slots: dict[str, Mano] = {}
    for i in orden:
        mano = candidatas[i]
        deseado = tabla.get(mano.etiqueta, defecto)
        if deseado not in slots:
            slots[deseado] = mano
        else:
            libre = "rh" if deseado == "lh" else "lh"
            slots[libre] = mano
    return slots


def frame_a_vector(manos: Sequence[Mano], ancho: int, alto: int) -> np.ndarray:
    """Vector de 126 float32 de un frame (reglas 3, 4 y 5). Sin manos da todo ceros."""
    vec = np.zeros(D, dtype=np.float32)
    for slot, mano in asignar_lateralidad(manos).items():
        inicio = 0 if slot == "lh" else LANDMARKS * 3
        vec[inicio:inicio + LANDMARKS * 3] = corregir_aspecto(mano.landmarks, ancho, alto).reshape(-1)
    return vec


def descartar_frames_vacios(secuencia: np.ndarray) -> np.ndarray:
    """Saca los frames sin ninguna mano (regla 6): todo ceros en los 126 valores."""
    sec = np.asarray(secuencia, dtype=np.float32).reshape(-1, D)
    return sec[(sec != 0).any(axis=1)]


def remuestrear_uniforme(secuencia: np.ndarray, t: int = T) -> np.ndarray:
    """Lleva n frames a t (regla 7): interpolación si n<t, submuestreo entero si n>t.

    La interpolación replica `SequenceBuffer.normalizeTo`: posiciones en float64 con
    linspace y el peso `w` pasado a float32 antes de mezclar.
    """
    sec = np.asarray(secuencia, dtype=np.float32).reshape(-1, D)
    n = len(sec)
    if n == 0:
        raise ValueError("secuencia sin frames válidos: no hay entrada para el modelo")
    if n == t:
        return sec.copy()
    if n > t:
        return sec[[(i * n) // t for i in range(t)]].copy()
    if n == 1:
        return np.repeat(sec, t, axis=0)
    paso = (n - 1) / (t - 1)
    out = np.empty((t, D), dtype=np.float32)
    for i in range(t):
        v = 0.0 + paso * i
        li, ui = int(np.floor(v)), int(np.ceil(v))
        if li == ui:
            out[i] = sec[li]
        else:
            w = np.float32(v - li)
            out[i] = (np.float32(1) - w) * sec[li] + w * sec[ui]
    return out


def secuencia_desde_frames(frames: Sequence[Sequence[Mano]], ancho: int, alto: int) -> np.ndarray:
    """Camino completo del contrato: frames crudos → entrada (T, 126) del modelo."""
    vectores = np.stack([frame_a_vector(f, ancho, alto) for f in frames]) if frames else np.zeros((0, D), np.float32)
    return remuestrear_uniforme(descartar_frames_vacios(vectores))


def secuencia_desde_vectores_v1(secuencia: np.ndarray) -> np.ndarray | None:
    """Adapta una secuencia v1 ya vectorizada (los .h5 de Holistic o el JSON del
    teléfono v1) a la regla temporal v2: descarta los vacíos y remuestrea.

    No corrige el aspecto ni la lateralidad: eso exige re-extraer los frames
    (PY-D4). Devuelve None si quedan menos de `MIN_FRAMES_VALIDOS` frames válidos,
    para que quien llama la excluya y la reporte.
    """
    validos = descartar_frames_vacios(secuencia)
    if len(validos) < MIN_FRAMES_VALIDOS:
        return None
    return remuestrear_uniforme(validos)
