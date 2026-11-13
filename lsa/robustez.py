"""Lógica del análisis de robustez de la sesión 2 (propuesta §3.5, PY-E5).

La sesión 2 grabó las 12 señas una vez en cada condición adversa (poca luz, cuerpo
girado 30°, cerca y lejos). Acá viven las piezas puras del análisis, separadas del
entrenamiento para poder testearlas: el nombre de cada condición, las correcciones
anotadas durante la grabación, las medidas de detección de manos por toma y los
resúmenes con intervalo de confianza.

Sobre los intervalos (auditoría, ML-M3 y ML-M6). Cada condición tiene 12 tomas de una,
dos o tres personas, y no todas las condiciones tienen a las tres. Por eso:

  - La caída de una condición se calcula persona por persona, contra el estándar de
    esa misma persona, y se promedia entre las personas que tienen la condición. Restar
    contra el estándar de las tres mezclaba el efecto de la condición con el de quién
    la grabó (ML-M6).
  - El intervalo remuestrea las tomas dentro de cada persona y también las semillas,
    porque el modelo de cada semilla es otra fuente de variación (ML-M3). Es un
    intervalo para ESTAS personas: con tres o menos no se puede estimar cuánto cambia
    el resultado de una persona a otra, y eso se informa aparte, persona por persona.
"""
from __future__ import annotations

import json
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



def matrices_por_persona(predicciones: Iterable[dict], condicion: str) -> dict[str, np.ndarray]:
    """Los aciertos de una condición, por persona: una matriz tomas × semillas de 0 y 1.
    Todas las tomas tienen que estar en todas las semillas (si falta una, es un error
    de la corrida y conviene que se note)."""
    por: dict[str, dict[str, dict[int, int]]] = {}
    for p in predicciones:
        if p["condicion"] == condicion:
            por.setdefault(p["persona"], {}).setdefault(p["id"], {})[p["semilla"]] = int(p["y"] == p["pred"])
    out = {}
    for persona, tomas in por.items():
        semillas = sorted({s for t in tomas.values() for s in t})
        out[persona] = np.array([[t[s] for s in semillas] for _, t in sorted(tomas.items())], dtype=float)
    return out


def _medias(m: np.ndarray, rng: np.random.Generator, semillas: np.ndarray) -> np.ndarray:
    """Una media por remuestreo: tomas elegidas con reposición, con las semillas dadas."""
    filas = rng.integers(0, len(m), (len(semillas), len(m)))
    return m[filas[:, :, None], semillas[:, None, :]].mean((1, 2))


def _semillas(matrices: Iterable[np.ndarray], rng: np.random.Generator, repeticiones: int) -> np.ndarray:
    n = {m.shape[1] for m in matrices}
    if len(n) != 1:
        raise ValueError(f"las matrices tienen distinta cantidad de semillas: {sorted(n)}")
    s = n.pop()
    return rng.integers(0, s, (repeticiones, s))


def acierto_con_ic(por_persona: dict[str, np.ndarray], repeticiones: int = 10000,
                   semilla: int = 0) -> tuple[float, float, float]:
    """Acierto de una condición (cada toma pesa lo mismo) con su IC 95 %: bootstrap de
    las tomas dentro de cada persona y de las semillas."""
    total = sum(len(m) for m in por_persona.values())
    media = sum(m.mean() * len(m) for m in por_persona.values()) / total
    rng = np.random.default_rng(semilla)
    sem = _semillas(por_persona.values(), rng, repeticiones)
    boot = sum(len(m) * _medias(m, rng, sem) for _, m in sorted(por_persona.items())) / total
    return (float(media), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def caida_pareada(ref: dict[str, np.ndarray], cond: dict[str, np.ndarray], repeticiones: int = 10000,
                  semilla: int = 0) -> dict | None:
    """Cuánto cae una condición respecto del estándar, persona por persona.

    Para cada persona que tiene la condición: su acierto en la condición menos su
    acierto estándar, con el mismo modelo. La caída es el promedio de esas diferencias
    (cada persona pesa lo mismo). Las semillas se remuestrean juntas para la
    referencia y la condición, porque es el mismo modelo el que evalúa las dos."""
    personas = sorted(p for p in cond if p in ref)
    if not personas:
        return None
    por_persona = {p: float(cond[p].mean() - ref[p].mean()) for p in personas}
    rng = np.random.default_rng(semilla)
    sem = _semillas([m for p in personas for m in (ref[p], cond[p])], rng, repeticiones)
    boot = sum(_medias(cond[p], rng, sem) - _medias(ref[p], rng, sem) for p in personas) / len(personas)
    return {"caida": float(np.mean(list(por_persona.values()))),
            "ic95": (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
            "referencia": float(np.mean([ref[p].mean() for p in personas])),
            "por_persona": por_persona}
