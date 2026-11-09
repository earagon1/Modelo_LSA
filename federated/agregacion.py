"""Agregación robusta: alternativas a FedAvg que resisten a un cliente que manda basura.

FedAvg (`protocolo.agregar`, `fedavg.aggregate`) promedia ponderando por n_k: un solo
update malo con norma grande, o con un n declarado enorme, mueve el promedio tanto como
quiera. Las dos estrategias de acá trabajan coordenada por coordenada y **no usan n_k**,
a propósito: n lo declara el cliente, y si pesara, el mismo ataque de inflar n volvería
a funcionar. El costo es que un cliente con muchas tomas pesa lo mismo que uno con pocas.

  - mediana: la mediana de los k valores de cada coordenada (Yin et al., 2018,
    "Byzantine-Robust Distributed Learning"). Con k = 3 y un atacante, se queda con un
    valor honesto. Con k = 2 es igual a la media: no defiende nada.
  - media recortada (beta): descarta los floor(beta·k) valores más altos y los más bajos
    de cada coordenada y promedia el resto (mismo trabajo). Con beta·k < 1 no descarta
    nada y es la media sin ponderar.

Trabajan sobre deltas (o sobre pesos: con w común da lo mismo, porque mediana(w + d_k) =
w + mediana(d_k)). Solo numpy: el servidor no carga TensorFlow.
"""
from __future__ import annotations

import numpy as np

ESTRATEGIAS = ("fedavg", "mediana", "recortada")
BETA_POR_DEFECTO = 0.2


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


def combinar(updates: list[tuple[list[np.ndarray], int]], estrategia: str,
             beta: float = BETA_POR_DEFECTO) -> list[np.ndarray]:
    """La agregación robusta pedida. "fedavg" no pasa por acá: lo hace `protocolo.agregar`,
    que lleva el promedio ponderado con la misma aritmética que `fedavg.aggregate`."""
    if estrategia == "mediana":
        return mediana(updates)
    if estrategia == "recortada":
        return media_recortada(updates, beta)
    raise ValueError(f"estrategia {estrategia!r}: las robustas son 'mediana' y 'recortada'")


def advertencia(estrategia: str, k: int, beta: float = BETA_POR_DEFECTO) -> str | None:
    """Por qué la estrategia no defiende nada con k clientes, o None si sí defiende."""
    if estrategia == "mediana" and k < 3:
        return f"mediana con {k} cliente(s) es igual a la media: hace falta un quórum de 3 o más"
    if estrategia == "recortada" and descartados_por_lado(k, beta) == 0:
        return (f"media recortada con beta={beta} y {k} cliente(s) no descarta nada: "
                f"hacen falta al menos {int(np.ceil(1 / beta)) if beta > 0 else '∞'} clientes")
    return None
