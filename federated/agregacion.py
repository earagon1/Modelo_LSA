"""Agregación robusta: alternativas a FedAvg que resisten a un cliente que manda basura.

FedAvg (`protocolo.agregar`, `fedavg.aggregate`) promedia ponderando por n_k: un solo
update malo con norma grande, o con un n declarado enorme, mueve el promedio tanto como
quiera. Las estrategias de acá **no usan n_k**, a propósito: n lo declara el cliente, y
si pesara, el mismo ataque de inflar n volvería a funcionar. El costo es que un cliente
con muchas tomas pesa lo mismo que uno con pocas.

  - mediana: la mediana de los k valores de cada coordenada (Yin et al., 2018,
    "Byzantine-Robust Distributed Learning"). Con k = 3 y un atacante, se queda con un
    valor honesto. Con k = 2 es igual a la media: no defiende nada.
  - media recortada (beta): descarta los floor(beta·k) valores más altos y los más bajos
    de cada coordenada y promedia el resto (mismo trabajo). Con beta·k < 1 no descarta
    nada y es la media sin ponderar.
  - Krum (f): elige el update entero (no coordenada por coordenada) que está más cerca
    de sus k − f − 2 vecinos (Blanchard et al., 2017, "Machine Learning with
    Adversaries"). Tolera hasta f atacantes si k ≥ 2f + 3: con el piloto de 2-3
    teléfonos no corre ni con f = 1 (pide 5). Es para la simulación con muchos clientes.
    Multi-Krum (m): promedia, sin ponderar, los m updates con mejor puntaje.

Las coordenada por coordenada trabajan igual sobre deltas que sobre pesos (con w común,
mediana(w + d_k) = w + mediana(d_k)); Krum también, porque las distancias entre
updates no cambian al sumarles el mismo w. Solo numpy: el servidor no carga TensorFlow.
"""
from __future__ import annotations

import numpy as np

ESTRATEGIAS = ("fedavg", "mediana", "recortada", "krum")
BETA_POR_DEFECTO = 0.2
F_POR_DEFECTO = 1
M_POR_DEFECTO = 1  # m = 1 es Krum; m > 1, Multi-Krum


def _apilar(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Un arreglo (k, *forma) por tensor, en float64."""
    if not updates:
        raise ValueError("no hay updates para agregar")
    largo = len(updates[0][0])
    if any(len(t) != largo for t, _ in updates):
        raise ValueError("los updates traen distinta cantidad de tensores")
    pilas = []
    for i in range(largo):
        formas = {np.shape(t[i]) for t, _ in updates}
        if len(formas) != 1:
            raise ValueError(f"el tensor {i} viene con formas distintas: {sorted(formas)}")
        pilas.append(np.stack([np.asarray(t[i], np.float64) for t, _ in updates]))
    return pilas


def descartados_por_lado(k: int, beta: float) -> int:
    """Cuántos valores saca la media recortada de cada punta, con k clientes."""
    if not 0 <= beta < 0.5:
        raise ValueError(f"beta tiene que estar en [0; 0,5): vino {beta}")
    return int(np.floor(beta * k))


def mediana(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """Mediana por coordenada. Ignora n_k (ver el docstring del módulo)."""
    return [np.median(p, axis=0).astype(np.float32) for p in _apilar(updates)]


def media_recortada(updates: list[tuple[list[np.ndarray], int]], beta: float = BETA_POR_DEFECTO) -> list[np.ndarray]:
    """Media por coordenada sin los floor(beta·k) extremos de cada lado. Ignora n_k."""
    k = len(updates)
    b = descartados_por_lado(k, beta)
    resultado = []
    for p in _apilar(updates):
        ordenado = np.sort(p, axis=0)
        resultado.append(ordenado[b:k - b].mean(axis=0).astype(np.float32))
    return resultado


def clientes_minimos_krum(f: int) -> int:
    """k ≥ 2f + 3: cada honesto tiene que tener más vecinos honestos que atacantes."""
    if isinstance(f, bool) or not isinstance(f, int) or f < 0:
        raise ValueError(f"f (atacantes tolerados) tiene que ser un entero ≥ 0: vino {f!r}")
    return 2 * f + 3


def validar_krum(k: int, f: int, m: int) -> str | None:
    """Por qué Krum no puede correr con k clientes, o None si puede."""
    minimo = clientes_minimos_krum(f)
    if isinstance(m, bool) or not isinstance(m, int) or m < 1:
        raise ValueError(f"m (updates que promedia Multi-Krum) tiene que ser un entero ≥ 1: vino {m!r}")
    if k < minimo:
        return f"Krum con f={f} necesita al menos {minimo} clientes y hay {k}"
    if m > k - f:
        return f"Multi-Krum con m={m} y f={f} necesita al menos {m + f} clientes y hay {k}"
    return None


def puntajes_krum(updates: list[tuple[list[np.ndarray], int]], f: int) -> np.ndarray:
    """Puntaje de cada update: la suma de las distancias al cuadrado a sus k − f − 2
    vecinos más cercanos (sin contarse a sí mismo). Menor = más central."""
    pilas = _apilar(updates)
    k = len(updates)
    planos = np.concatenate([p.reshape(k, -1) for p in pilas], axis=1)
    cuadrados = np.sum(planos ** 2, axis=1)
    # |a − b|² = |a|² + |b|² − 2·a·b, sin armar k·k vectores de diferencias.
    distancias = np.maximum(cuadrados[:, None] + cuadrados[None, :] - 2 * planos @ planos.T, 0.0)
    vecinos = k - f - 2
    return np.array([np.sort(np.delete(distancias[i], i))[:vecinos].sum() for i in range(k)])


def elegidos_krum(updates: list[tuple[list[np.ndarray], int]], f: int = F_POR_DEFECTO,
                  m: int = M_POR_DEFECTO) -> list[int]:
    """Índices de los m updates con menor puntaje (en empate, el que llegó primero)."""
    motivo = validar_krum(len(updates), f, m)
    if motivo:
        raise ValueError(motivo)
    return [int(i) for i in np.argsort(puntajes_krum(updates, f), kind="stable")[:m]]


def krum(updates: list[tuple[list[np.ndarray], int]], f: int = F_POR_DEFECTO,
         m: int = M_POR_DEFECTO) -> tuple[list[np.ndarray], list[int]]:
    """(promedio sin ponderar de los m updates con menor puntaje, sus índices). Ignora n_k."""
    elegidos = elegidos_krum(updates, f, m)
    pilas = _apilar(updates)
    return [p[elegidos].mean(axis=0).astype(np.float32) for p in pilas], elegidos


def combinar(updates: list[tuple[list[np.ndarray], int]], estrategia: str, beta: float = BETA_POR_DEFECTO,
             f: int = F_POR_DEFECTO, m: int = M_POR_DEFECTO) -> list[np.ndarray]:
    """La agregación robusta pedida. "fedavg" no pasa por acá: lo hace `protocolo.agregar`,
    que lleva el promedio ponderado con la misma aritmética que `fedavg.aggregate`."""
    if estrategia == "mediana":
        return mediana(updates)
    if estrategia == "recortada":
        return media_recortada(updates, beta)
    if estrategia == "krum":
        return krum(updates, f, m)[0]
    raise ValueError(f"estrategia {estrategia!r}: las robustas son 'mediana', 'recortada' y 'krum'")


def advertencia(estrategia: str, k: int, beta: float = BETA_POR_DEFECTO, f: int = F_POR_DEFECTO,
                m: int = M_POR_DEFECTO) -> str | None:
    """Por qué la estrategia no defiende nada (o, Krum, no puede correr) con k clientes."""
    if estrategia == "mediana" and k < 3:
        return f"mediana con {k} cliente(s) es igual a la media: hace falta un quórum de 3 o más"
    if estrategia == "recortada" and descartados_por_lado(k, beta) == 0:
        return (f"media recortada con beta={beta} y {k} cliente(s) no descarta nada: "
                f"hacen falta al menos {int(np.ceil(1 / beta)) if beta > 0 else '∞'} clientes")
    if estrategia == "krum":
        return validar_krum(k, f, m)
    return None
