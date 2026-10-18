"""Lógica del análisis de robustez de la sesión 2 (propuesta §3.5, PY-E5).

La sesión 2 grabó las 12 señas una vez en cada condición adversa (poca luz, cuerpo
girado 30°, cerca y lejos). Acá viven las piezas puras del análisis, separadas del
entrenamiento para poder testearlas: el nombre de cada condición, las correcciones
anotadas durante la grabación, las medidas de detección de manos por toma y los
resúmenes con intervalo de confianza.
"""
from __future__ import annotations

import json
import math
from collections.abc import Iterable

import numpy as np

ESTANDAR = "estándar"
# Orden de presentación en las tablas: de la referencia a lo más adverso.
CONDICIONES = [ESTANDAR, "poca luz", "cerca", "lejos", "lejos + poca luz", "girado 30°"]
# Señas que se hacen con las dos manos: en ellas importa si se detectaron ambas.
DOS_MANOS = ("por favor", "como estas")


def nombre_condicion(condicion: str) -> str:
    """"luz=sombra;angulo=0;distancia=media" -> "poca luz". La estándar es luz interior,
    de frente y a distancia media; cada factor adverso suma su nombre."""
    f = dict(p.split("=", 1) for p in condicion.split(";") if "=" in p)
    partes = []
    if f.get("distancia") in ("cerca", "lejos"):
        partes.append(f["distancia"])
    if f.get("luz") == "sombra":
        partes.append("poca luz")
    elif f.get("luz") == "exterior":
        partes.append("luz natural")
    if f.get("angulo", "0") != "0":
        partes.append(f"girado {f['angulo']}°")
    return " + ".join(partes) if partes else ESTANDAR


def leer_correcciones(ruta: str) -> dict[str, list[tuple[int, int, str]]]:
    """Las condiciones reales anotadas durante la sesión (`correcciones_sesion2.json`):
    por persona, rangos de órdenes [desde, hasta] con la condición que hubo de verdad.
    Los crudos no se editan; el análisis aplica estas anotaciones."""
    with open(ruta, encoding="utf-8") as f:
        d = json.load(f)
    out: dict[str, list[tuple[int, int, str]]] = {}
    for persona, c in d.items():
        if not isinstance(c, dict):
            continue
        for r in c.get("condicion_real", []):
            desde, hasta = r["ordenes"]
            out.setdefault(persona, []).append((int(desde), int(hasta), r["real"]))
    return out


def condicion_efectiva(persona: str, orden: int, bloque: str, condicion: str,
                       correcciones: dict[str, list[tuple[int, int, str]]]) -> str:
    """Nombre de la condición en la que se grabó la toma, con las correcciones aplicadas.
    Las tandas de la sesión 1 (A) son la referencia estándar."""
    if bloque != "B":
        return ESTANDAR
    for desde, hasta, real in correcciones.get(persona, []):
        if desde <= orden <= hasta:
            return nombre_condicion(real)
    return nombre_condicion(condicion)


def deteccion(muestra: dict) -> dict:
    """Qué vio HandLandmarker en la toma: fracción de frames con alguna mano y, en las
    señas de dos manos, fracción con las dos. Mide el extractor, no el modelo."""
    frames = muestra["frames"]
    n = len(frames)
    con_mano = sum(1 for f in frames if f["manos"]) / n if n else 0.0
    ambas = sum(1 for f in frames if len(f["manos"]) >= 2) / n if n else 0.0
    return {"frames": n, "con_mano": con_mano,
            "ambas": ambas if muestra["label"] in DOS_MANOS else None, "fps": float(muestra.get("fps", 0.0))}


def wilson(aciertos: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson al 95 % para una proporción. Con 12 tomas por persona y
    condición, el intervalo normal se sale de [0, 1]; Wilson no."""
    if n == 0:
        return (math.nan, math.nan)
    p = aciertos / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    radio = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centro - radio), min(1.0, centro + radio))


def acierto_por_muestra(predicciones: Iterable[dict]) -> dict[str, float]:
    """Por toma (clave `id`), la fracción de semillas que la acertó. Cada toma pesa lo
    mismo aunque haya varias semillas: el n del intervalo es el de tomas, no el de
    tomas × semillas."""
    suma: dict[str, list[int]] = {}
    for p in predicciones:
        suma.setdefault(p["id"], []).append(int(p["y"] == p["pred"]))
    return {k: float(np.mean(v)) for k, v in suma.items()}


def diferencia_bootstrap(a: np.ndarray, b: np.ndarray, repeticiones: int = 10000,
                         semilla: int = 0) -> tuple[float, float, float]:
    """Diferencia de acierto medio b - a con IC 95 % por bootstrap sobre las tomas.
    Las dos muestras son tomas distintas (no pareadas), así que se remuestrea cada una
    por separado."""
    rng = np.random.default_rng(semilla)
    if len(a) == 0 or len(b) == 0:
        return (math.nan, math.nan, math.nan)
    ia = rng.integers(0, len(a), (repeticiones, len(a)))
    ib = rng.integers(0, len(b), (repeticiones, len(b)))
    d = b[ib].mean(1) - a[ia].mean(1)
    return (float(b.mean() - a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
