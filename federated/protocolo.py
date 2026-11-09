"""Protocolo del federado por red local (contrato `contract/federado_v1.json`).

Lo que viaja entre el teléfono y el servidor, y las cuentas que tienen que dar igual
en los dos lados:
  - los pesos de la cabeza (float32 little-endian en base64) y su sha256;
  - el delta del cliente cuantizado a 8 bits con redondeo estocástico. Para que
    Kotlin y Python den lo mismo bit a bit, los uniformes del redondeo se inyectan
    (el fixture los trae fijos): los generadores de números al azar de los dos
    lenguajes no producen la misma secuencia;
  - la rotación aleatoria antes de cuantizar (opcional, "actualizaciones esbozadas"
    de Konečný et al., que cita la propuesta): x -> H·D·x / sqrt(n), con D signos al
    azar de una semilla que viaja en el update y H la transformada de Walsh-Hadamard.
    Reparte la energía del delta entre todos los valores, así un valor grande no fija
    la escala de todo el tensor y el error de los 8 bits baja. El servidor la deshace;
    no hace falta paridad bit a bit (los signos sí: salen del mismo splitmix64);
  - la agregación: w_{r+1} = w_r + sum_k (n_k / n) * delta_k, que es FedAvg
    (`fedavg.aggregate`) escrito sobre deltas. Va acá y no se importa de fedavg.py
    porque ese módulo carga TensorFlow, y el servidor no lo necesita.

Sin dependencias fuera de numpy: el servidor corre en cualquier PC.
"""
from __future__ import annotations

import base64
import hashlib

import agregacion
import numpy as np

PROTOCOLO = "federado-v1"
BITS = 8
NIVELES = 2 ** (BITS - 1) - 1  # simétrico: [-127, 127]
ROTACION = "hadamard"
_MASCARA = (1 << 64) - 1

# Tope al peso de un cliente en FedAvg. n_muestras lo declara el cliente y es el peso del
# promedio: sin tope, un update que diga 10⁶ muestras se queda con casi todo el promedio,
# y el límite de norma no lo frena (acota cuánto se mueve el delta, no cuánto pesa). Lo
# encontró la auditoría del 2026-11-09 (X5). Pasarlo no es rechazo: el update entra con
# peso N_MAX_MUESTRAS y queda registrado (decisión D-R3, plan/federado_robustez.md).
#
# Ojo con lo que el tope NO hace: en el piloto cada persona tiene unas 106 tomas en el
# teléfono (las ~1090 de la webcam de P1 nunca pasan por el federado), así que con 2000
# un atacante contra dos honestos todavía pesa el 90 % (2000 / 2212). El tope solo frena
# el 10⁶; contra un atacante la defensa es la agregación robusta (agregacion.py), que no
# usa n. Bajarlo cerca del máximo honesto (p. ej. 200 → 49 %) castiga a quien junte
# más sesiones: queda como decisión abierta en plan/federado_robustez.md.
N_MAX_MUESTRAS = 2000


def potencia_de_2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def signos(semilla: int, n: int) -> np.ndarray:
    """n signos ±1 de splitmix64 (el mismo generador en Kotlin): el i-ésimo usa el estado
    semilla + (i+1)·0x9E3779B97F4A7C15 y toma el bit más alto del resultado."""
    with np.errstate(over="ignore"):
        z = np.uint64(semilla & _MASCARA) + np.arange(1, n + 1, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return np.where((z >> np.uint64(63)) == 0, 1.0, -1.0)


def _hadamard(x: np.ndarray) -> np.ndarray:
    """Transformada de Walsh-Hadamard rápida (sin normalizar); len(x) potencia de 2."""
    y = np.array(x, dtype=np.float64)
    h = 1
    while h < len(y):
        y = y.reshape(-1, 2, h)
        y = np.stack([y[:, 0] + y[:, 1], y[:, 0] - y[:, 1]], axis=1).reshape(-1)
        h *= 2
    return y


def rotar(x: np.ndarray, semilla: int) -> np.ndarray:
    """H·D·x / sqrt(n), con x completado con ceros hasta n = potencia de 2. Es ortogonal:
    conserva la norma, y [desrotar] la deshace."""
    plano = np.asarray(x, np.float64).ravel()
    n = potencia_de_2(plano.size)
    relleno = np.zeros(n)
    relleno[:plano.size] = plano
    return (_hadamard(signos(semilla, n) * relleno) / np.sqrt(n)).astype(np.float32)


def desrotar(y: np.ndarray, semilla: int, largo: int) -> np.ndarray:
    n = len(y)
    return (signos(semilla, n) * _hadamard(np.asarray(y, np.float64)) / np.sqrt(n))[:largo].astype(np.float32)


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


def codificar_delta(delta: list[np.ndarray], rng: np.random.Generator, semilla: int | None = None) -> list[dict]:
    """Con [semilla], cada tensor i se rota con semilla + i antes de cuantizar."""
    salida = []
    for i, d in enumerate(delta):
        t = {"forma": list(d.shape)}
        x = d
        if semilla is not None:
            x = rotar(d, semilla + i)
            t["rotacion"] = {"tipo": ROTACION, "semilla": semilla + i}
        q, escala = cuantizar(x, rng.random(x.shape, dtype=np.float32))
        t.update(escala=escala, datos=a_b64(q, np.int8))
        salida.append(t)
    return salida


def _largo_enviado(t: dict) -> int:
    n = int(np.prod(t["forma"]))
    return potencia_de_2(n) if t.get("rotacion") else n


def decodificar_delta(tensores: list[dict]) -> list[np.ndarray]:
    salida = []
    for t in tensores:
        n = int(np.prod(t["forma"]))
        rot = t.get("rotacion")
        if rot is None:
            salida.append(descuantizar(de_b64(t["datos"], np.int8, t["forma"]), t["escala"]))
            continue
        if rot.get("tipo") != ROTACION:
            raise ValueError(f"rotación {rot.get('tipo')!r} desconocida")
        y = descuantizar(de_b64(t["datos"], np.int8, [potencia_de_2(n)]), t["escala"])
        salida.append(desrotar(y, int(rot["semilla"]), n).reshape(t["forma"]))
    return salida


def bytes_payload(tensores: list[dict]) -> int:
    """Bytes útiles del delta (sin base64 ni JSON): 1 por valor enviado + 4 por escala
    (+ 8 por la semilla si va rotado)."""
    return sum(_largo_enviado(t) + 4 + (8 if t.get("rotacion") else 0) for t in tensores)


def norma(delta: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(np.asarray(d, np.float64) ** 2)) for d in delta)))


def recortar(delta: list[np.ndarray], maxima: float | None) -> tuple[list[np.ndarray], float, bool]:
    """Límite a la magnitud del update (tabla de riesgos de la propuesta: "limitar la
    magnitud de actualizaciones"). Si la norma L2 del delta completo pasa [maxima], se lo
    escala hasta [maxima]: la dirección se conserva, pero un cliente no puede mover la
    cabeza más que eso en una ronda. Devuelve (delta, norma original, si se recortó)."""
    n = norma(delta)
    if maxima is None or n <= maxima or n == 0:
        return delta, n, False
    f = maxima / n
    return [(np.asarray(d, np.float32) * np.float32(f)).astype(np.float32) for d in delta], n, True


def peso_efectivo(n: int, maximo: int | None = N_MAX_MUESTRAS) -> tuple[int, bool]:
    """(peso con el que entra el update al promedio, si se recortó). maximo=None: sin tope."""
    if maximo is None or n <= maximo:
        return n, False
    return maximo, True


def validar_update(u: dict, ronda: int, formas: list[list[int]]) -> str | None:
    """Motivo de rechazo, o None si el update es aceptable para esta ronda."""
    if u.get("protocolo") != PROTOCOLO:
        return f"protocolo {u.get('protocolo')!r}, se esperaba {PROTOCOLO!r}"
    if u.get("ronda") != ronda:
        return f"ronda {u.get('ronda')}: la actual es {ronda}"
    if not isinstance(u.get("participant_id"), str) or not u["participant_id"]:
        return "falta participant_id"
    n = u.get("n_muestras")
    # bool es subclase de int en Python: sin este chequeo, `true` pasaba como 1 muestra.
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
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


def agregar(pesos: list[np.ndarray], updates: list[tuple[list[np.ndarray], int]],
            estrategia: str = "fedavg", **parametros) -> list[np.ndarray]:
    """w + sum_k (n_k / n) * delta_k, acumulado en float64 como `fedavg.aggregate`.
    Con estrategia "mediana", "recortada" o "krum": w + la combinación robusta de los
    deltas (`agregacion.py`, que no usa n_k; `parametros` son beta, f y m)."""
    if not updates:
        raise ValueError("no hay updates para agregar")
    if estrategia != "fedavg":
        combinado = agregacion.combinar(updates, estrategia, **parametros)
        return [(np.asarray(w, np.float64) + np.asarray(c, np.float64)).astype(np.float32)
                for w, c in zip(pesos, combinado, strict=True)]
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
