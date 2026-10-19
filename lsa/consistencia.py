"""Métrica de consistencia estadística σ de la propuesta (sección de evaluación).

Hipótesis: entradas parecidas tienen que dar salidas parecidas, y eso se puede medir
sin etiquetas (la app se usa en tiempo real, donde no hay etiquetas). Definición
computacional, tal como la escribe la propuesta:

  1. Entradas {X_i} (la secuencia que recibe el modelo, 15×126) y salidas {ŷ_i} ∈ R^C
     (probabilidades del modelo).
  2. Umbral ε > 0: X_i y X_j son vecinas si ||X_i − X_j|| < ε (distancia euclídea).
  3. N_i = {j : ||X_i − X_j|| < ε}.
  4. σ_i = mean_c( std_sesgada({ŷ_jc : j ∈ N_i}) ), con std_sesgada = sqrt((1/m) Σ (ŷ_jc − ȳ_c)²),
     m = |N_i|.
  5. σ = (1/|I|) Σ_{i∈I} σ_i, con I = {i : |N_i| > 1}.

Dos decisiones que la propuesta deja abiertas, y cómo se toman acá:
  - N_i incluye a i (la distancia a sí misma es 0 < ε), como dice la definición del paso
    3. Por eso |N_i| > 1 equivale a tener al menos otro vecino: es la condición de "al
    menos dos vecinos para que tenga sentido la desviación estándar".
  - ε no está fijado: se toma de los datos como un percentil de las distancias entre
    pares ([epsilon_por_percentil]) y se informa σ para varios, así la conclusión no
    depende de un número elegido a mano.
"""
from __future__ import annotations

import numpy as np


def distancias(X: np.ndarray) -> np.ndarray:
    """Matriz de distancias euclídeas entre las entradas aplanadas."""
    A = np.asarray(X, np.float64).reshape(len(X), -1)
    cuadrados = np.sum(A * A, axis=1)
    d2 = cuadrados[:, None] + cuadrados[None, :] - 2.0 * (A @ A.T)
    np.fill_diagonal(d2, 0.0)
    return np.sqrt(np.maximum(d2, 0.0))


def epsilon_por_percentil(D: np.ndarray, percentil: float) -> float:
    """El percentil [percentil] de las distancias entre pares distintos (i < j)."""
    i, j = np.triu_indices(len(D), k=1)
    return float(np.percentile(D[i, j], percentil))


def sigma(P: np.ndarray, D: np.ndarray, epsilon: float) -> dict:
    """σ de la propuesta para salidas P (n×C) con distancias D (n×n) y umbral ε.

    Devuelve σ, los σ_i de las muestras de I, |I| y el tamaño medio de las vecindades.
    σ es None si ninguna muestra tiene vecinos (ε demasiado chico)."""
    P = np.asarray(P, np.float64)
    vecinos = D < epsilon
    sigmas = np.full(len(P), np.nan)
    tamanos = vecinos.sum(axis=1)
    for i in np.where(tamanos > 1)[0]:
        sigmas[i] = float(np.mean(np.std(P[vecinos[i]], axis=0)))  # np.std usa 1/m: la sesgada
    en_I = ~np.isnan(sigmas)
    return {"sigma": float(np.mean(sigmas[en_I])) if en_I.any() else None,
            "sigmas_i": sigmas, "en_I": en_I, "n_I": int(en_I.sum()), "n": len(P),
            "vecinos_medio": float(np.mean(tamanos[en_I])) if en_I.any() else 0.0}


def pureza(D: np.ndarray, y: np.ndarray, epsilon: float) -> float | None:
    """Fracción de pares de vecinos distintos (i != j, distancia < ε) que son la misma seña.

    No es parte de σ (σ no usa etiquetas): sirve para elegir ε. La propuesta quiere que
    los vecinos sean "el mismo gesto con pequeñas variaciones"; si ε es grande, entran
    señas distintas y σ pasa a medir cuánto se mezclan las clases."""
    vecinos = (D < epsilon) & ~np.eye(len(D), dtype=bool)
    if not vecinos.any():
        return None
    i, j = np.where(vecinos)
    return float(np.mean(np.asarray(y)[i] == np.asarray(y)[j]))
