"""Lector del dataset v2 del teléfono (`lsa_samples_v2.jsonl`, T-A14 / PY-D5).

Cada línea es una muestra con los landmarks crudos de cada frame, tal como los
entregó HandLandmarker en la app (`dataset/MuestraV2.kt`). Acá se convierten a la
entrada del modelo con el contrato v2 (`lsa.contrato`), así el mismo archivo sirve
para cualquier versión futura del contrato sin volver a grabar.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from lsa import contrato
from lsa.contrato import Mano

ESQUEMA = "senar-muestra-v2"


def leer(ruta: str | Path) -> Iterator[dict]:
    """Muestras del archivo, en orden. Una línea que no es JSON válido corta con el
    número de línea: un archivo truncado no se tiene que usar en silencio."""
    with open(ruta, encoding="utf-8") as f:
        for n, linea in enumerate(f, 1):
            if not linea.strip():
                continue
            try:
                muestra = json.loads(linea)
            except json.JSONDecodeError as e:
                raise ValueError(f"{ruta}:{n}: línea inválida ({e})") from e
            if muestra.get("esquema") != ESQUEMA:
                raise ValueError(f"{ruta}:{n}: esquema {muestra.get('esquema')!r}, se esperaba {ESQUEMA!r}")
            yield muestra


def frames_como_manos(muestra: dict) -> list[list[Mano]]:
    """Los frames de la muestra en el formato que usa `lsa.contrato`."""
    return [[Mano(m["e"], float(m["s"]), np.asarray(m["lm"], np.float32).reshape(21, 3)) for m in fr["manos"]]
            for fr in muestra["frames"]]


def secuencia(muestra: dict) -> np.ndarray:
    """Entrada (T, 126) del modelo según el contrato v2."""
    return contrato.secuencia_desde_frames(frames_como_manos(muestra), muestra["img_w"], muestra["img_h"])
