"""Protocolo del federado por red local (contrato `contract/federado_v1.json`).

Lo que viaja entre el teléfono y el servidor, y las cuentas que tienen que dar igual
en los dos lados:
  - los pesos de la cabeza (float32 little-endian en base64) y su sha256;
  - el delta del cliente cuantizado a 8 bits con redondeo estocástico. Para que
    Kotlin y Python den lo mismo bit a bit, los uniformes del redondeo se inyectan
    (el fixture los trae fijos): los generadores de números al azar de los dos
    lenguajes no producen la misma secuencia;
  - la agregación: w_{r+1} = w_r + sum_k (n_k / n) * delta_k, que es FedAvg
    (`fedavg.aggregate`) escrito sobre deltas. Va acá y no se importa de fedavg.py
    porque ese módulo carga TensorFlow, y el servidor no lo necesita.

Sin dependencias fuera de numpy: el servidor corre en cualquier PC.
"""
from __future__ import annotations

import base64
import hashlib

import numpy as np

PROTOCOLO = "federado-v1"
BITS = 8
NIVELES = 2 ** (BITS - 1) - 1  # simétrico: [-127, 127]


def a_b64(x: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode("ascii")


def de_b64(s: str, dtype, forma) -> np.ndarray:
    datos = np.frombuffer(base64.b64decode(s), dtype=np.dtype(dtype).newbyteorder("<"))
    if datos.size != int(np.prod(forma)):
        raise ValueError(f"el tensor trae {datos.size} valores y la forma {list(forma)} pide {int(np.prod(forma))}")
    return datos.astype(dtype).reshape(forma)


def sha_pesos(pesos: list[np.ndarray]) -> str:
    """sha256 de los tensores en orden, como float32 little-endian concatenados.
    La app lo recalcula al descargar: si no coincide, no usa la cabeza."""
    h = hashlib.sha256()
    for w in pesos:
        h.update(np.ascontiguousarray(w, dtype="<f4").tobytes())
    return h.hexdigest()


def codificar_pesos(pesos: list[np.ndarray]) -> list[dict]:
    return [{"forma": list(w.shape), "datos": a_b64(w, np.float32)} for w in pesos]


def decodificar_pesos(tensores: list[dict]) -> list[np.ndarray]:
    return [de_b64(t["datos"], np.float32, t["forma"]) for t in tensores]


def cuantizar(x: np.ndarray, uniformes: np.ndarray) -> tuple[np.ndarray, float]:
    """Cuantización simétrica por tensor a 8 bits con redondeo estocástico.

    Igual que `compresion.cuantizar(x, 8, rng)`, pero con los uniformes U[0,1) dados:
    sube al entero siguiente si u < parte fraccionaria. Es insesgado (E[q] = x/escala).
    La escala va en float32 porque así viaja; las cuentas se hacen en float32 para
    que Kotlin pueda repetirlas exactas."""
    x = np.asarray(x, np.float32)
    maximo = np.float32(np.max(np.abs(x))) if x.size else np.float32(0)
    if maximo == 0:
        return np.zeros(x.shape, np.int8), 0.0
    escala = np.float32(maximo / np.float32(NIVELES))
    escalado = (x / escala).astype(np.float32)
    piso = np.floor(escalado)
    q = piso + (np.asarray(uniformes, np.float32).reshape(x.shape) < (escalado - piso))
    return np.clip(q, -NIVELES, NIVELES).astype(np.int8), float(escala)


def descuantizar(q: np.ndarray, escala: float) -> np.ndarray:
    return (q.astype(np.float32) * np.float32(escala)).astype(np.float32)


def codificar_delta(delta: list[np.ndarray], rng: np.random.Generator) -> list[dict]:
    salida = []
    for d in delta:
        q, escala = cuantizar(d, rng.random(d.shape, dtype=np.float32))
        salida.append({"forma": list(d.shape), "escala": escala, "datos": a_b64(q, np.int8)})
    return salida


def decodificar_delta(tensores: list[dict]) -> list[np.ndarray]:
    return [descuantizar(de_b64(t["datos"], np.int8, t["forma"]), t["escala"]) for t in tensores]


def bytes_payload(tensores: list[dict]) -> int:
    """Bytes útiles del delta (sin base64 ni JSON): 1 por valor + 4 por escala."""
    return sum(int(np.prod(t["forma"])) + 4 for t in tensores)


def validar_update(u: dict, ronda: int, formas: list[list[int]]) -> str | None:
    """Motivo de rechazo, o None si el update es aceptable para esta ronda."""
    if u.get("protocolo") != PROTOCOLO:
        return f"protocolo {u.get('protocolo')!r}, se esperaba {PROTOCOLO!r}"
    if u.get("ronda") != ronda:
        return f"ronda {u.get('ronda')}: la actual es {ronda}"
    if not isinstance(u.get("participant_id"), str) or not u["participant_id"]:
        return "falta participant_id"
    n = u.get("n_muestras")
    if not isinstance(n, int) or n <= 0:
        return "n_muestras tiene que ser un entero positivo"
    if u.get("bits") != BITS:
        return f"bits {u.get('bits')}: el protocolo usa {BITS}"
    tensores = u.get("tensores")
    if not isinstance(tensores, list) or [t.get("forma") for t in tensores] != formas:
        return "las formas de los tensores no son las de la cabeza"
    try:
        delta = decodificar_delta(tensores)
    except (ValueError, KeyError, TypeError) as e:
        return f"tensores ilegibles: {e}"
    if not all(np.isfinite(d).all() for d in delta):
        return "el delta trae NaN o infinitos"
    return None


def agregar(pesos: list[np.ndarray], updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """w + sum_k (n_k / n) * delta_k, acumulado en float64 como `fedavg.aggregate`."""
    if not updates:
        raise ValueError("no hay updates para agregar")
    total = sum(n for _, n in updates)
    acumulado = [np.asarray(w, np.float64).copy() for w in pesos]
    for delta, n in updates:
        for i, d in enumerate(delta):
            acumulado[i] += (n / total) * np.asarray(d, np.float64)
    return [a.astype(np.float32) for a in acumulado]


# --- la cabeza: 128 -> 64 relu -> 64 relu -> C softmax (igual que CabezaDensa.kt) ---

def cabeza_desde_json(d: dict) -> tuple[list[np.ndarray], list[str]]:
    """Pesos [W1, b1, W2, b2, W3, b3] y clases desde el JSON de `exportar_cabeza.py`."""
    pesos = []
    for c in d["capas"]:
        pesos += [np.asarray(c["W"], np.float32), np.asarray(c["b"], np.float32)]
    return pesos, list(d["clases"])


def predecir(pesos: list[np.ndarray], emb: np.ndarray) -> np.ndarray:
    x = np.asarray(emb, np.float32)
    for i in range(0, len(pesos), 2):
        x = x @ pesos[i] + pesos[i + 1]
        if i + 2 < len(pesos):
            x = np.maximum(x, 0)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)
