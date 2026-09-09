"""
Compresion de las actualizaciones que los clientes le mandan al servidor.

FedAvg tal cual se implemento en `fedavg.py` manda los pesos completos en
float32: en la corrida de 5 clientes eso dieron 193 MB para 30 rondas. Para un
escenario movil, que es el de este proyecto, es muchisimo.

Konecny et al. (2017) atacan exactamente ese problema y proponen dos familias
de tecnicas: actualizaciones *estructuradas* (restringir el update a bajo rango
o esparso) y actualizaciones *esbozadas* (transformar y cuantizar antes de
enviar). Aca se implementa la segunda, que es la que aplica sin tocar la
arquitectura del modelo.

QUE SE MANDA
------------
En lugar del peso `w_k`, el cliente manda el delta `d_k = w_k - w_global`.
El servidor ya tiene `w_global`, asi que con los deltas reconstruye lo mismo:

    w_nuevo = w_global + SUM (n_k / n) * d_k

Es algebraicamente identico a promediar pesos, y lo verifica
`test_compresion.py::test_deltas_equivalen_a_promediar_pesos`. La ventaja es
que el delta es chico y esta centrado en cero, asi que se cuantiza mucho mejor
que el peso absoluto.

REDONDEO ESTOCASTICO
--------------------
Al cuantizar hay que elegir como redondear, y no da igual:

  - Determinista (round a lo mas cercano) introduce un sesgo sistematico. Ese
    sesgo NO se cancela al promediar entre clientes: se acumula ronda tras
    ronda y desvia el modelo global.

  - Estocastico redondea para arriba o para abajo con probabilidad
    proporcional a la distancia, de modo que el valor esperado es exactamente
    el original. El error queda con media cero y, al promediar sobre K
    clientes, se cancela. Es la razon por la que Konecny et al. lo usan.

SOBRE LA MEDICION DE BYTES
--------------------------
Los valores se siguen guardando en float32: no se empaquetan de verdad en 4 u
8 bits. Lo que se simula es el COSTO de comunicacion, que se calcula
analiticamente con `bytes_de_update`. Es la practica habitual para este tipo
de experimento, y hay que decirlo en el informe: se mide cuanto se ahorraria,
no se implementa el protocolo de transporte.
"""

from __future__ import annotations

import numpy as np


def cuantizar(x: np.ndarray, bits: int, rng: np.random.Generator | None = None):
    """
    Cuantizacion uniforme por tensor, simetrica alrededor de cero.

    Devuelve (enteros, escala). Con `rng` el redondeo es estocastico (insesgado);
    sin `rng`, determinista.

    La escala se deriva del maximo absoluto del tensor, asi que cada tensor
    tiene la suya. Es lo minimo razonable: las capas tienen rangos muy
    distintos y una escala global desperdiciaria casi todos los niveles.
    """
    if bits < 2:
        raise ValueError("bits debe ser >= 2")

    niveles = 2 ** (bits - 1) - 1  # simetrico: [-niveles, +niveles]
    maximo = float(np.max(np.abs(x)))

    if maximo == 0.0:
        return np.zeros_like(x, dtype=np.float32), 0.0

    escala = maximo / niveles
    escalado = x / escala

    if rng is None:
        q = np.round(escalado)
    else:
        # Redondeo estocastico: la parte fraccionaria es la probabilidad de
        # subir al entero siguiente. E[q] == escalado, o sea insesgado.
        piso = np.floor(escalado)
        q = piso + (rng.random(escalado.shape) < (escalado - piso))

    return np.clip(q, -niveles, niveles).astype(np.float32), escala


def descuantizar(q: np.ndarray, escala: float) -> np.ndarray:
    return (q * escala).astype(np.float32)


def bytes_de_update(tensores: list[np.ndarray], bits: int) -> int:
    """
    Cuanto ocuparia mandar este update con `bits` por valor.

    Se suma 4 bytes por tensor para la escala, que viaja en float32. Con
    bits=32 devuelve el tamano float32 sin overhead, que es el caso base.
    """
    if bits >= 32:
        return int(sum(t.size for t in tensores) * 4)
    total_bits = sum(t.size for t in tensores) * bits
    return int(np.ceil(total_bits / 8)) + 4 * len(tensores)


def comprimir_update(
    delta: list[np.ndarray],
    bits: int,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], int]:
    """
    Aplica la cuantizacion a un delta completo y devuelve (delta_reconstruido, bytes).

    El delta reconstruido es lo que el servidor recibiria despues de
    descuantizar: con `bits >= 32` es el original sin tocar.
    """
    n_bytes = bytes_de_update(delta, bits)

    if bits >= 32:
        return [d.copy() for d in delta], n_bytes

    salida = []
    for t in delta:
        q, escala = cuantizar(t, bits, rng)
        salida.append(descuantizar(q, escala))
    return salida, n_bytes


def error_relativo(original: list[np.ndarray], reconstruido: list[np.ndarray]) -> float:
    """Norma del error sobre norma del original, sobre todos los tensores juntos."""
    num = sum(float(np.sum((a - b) ** 2)) for a, b in zip(original, reconstruido))
    den = sum(float(np.sum(a ** 2)) for a in original)
    return float(np.sqrt(num / den)) if den > 0 else 0.0
