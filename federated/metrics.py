"""
Metricas de evaluacion: las clasicas y la de consistencia estadistica.

Las clasicas (accuracy, F1 macro y micro, matriz de confusion) son las que
pide la seccion 3.5 de la propuesta y no estaban implementadas en ningun lado.

`consistency_sigma` implementa la metrica de consistencia estadistica definida
en las paginas 5 a 7 de la propuesta, siguiendo el procedimiento al pie:

    1. Vecindad de cada entrada:   N_i = { j : ||X_i - X_j|| < eps }
    2. Desvio sesgado por clase sobre esa vecindad (divide por m, no por m-1)
    3. sigma_i = promedio de esos desvios sobre las C clases
    4. sigma   = promedio de sigma_i sobre los i con |N_i| > 1

Un sigma bajo significa que entradas parecidas producen salidas parecidas, o
sea que el modelo es estable frente a variaciones sutiles del gesto. Es la
propiedad que accuracy y F1 no capturan: un modelo puede acertar mucho y aun
asi responder erraticamente ante dos ejecuciones casi identicas de la misma
sena.
"""

from __future__ import annotations

import numpy as np


def classification_metrics(y_true, probs) -> dict:
    """Accuracy, F1 macro y F1 micro a partir del vector de probabilidades."""
    from sklearn.metrics import accuracy_score, f1_score

    y_pred = np.argmax(probs, axis=1)
    y_true = np.asarray(y_true)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
    }


def matriz_confusion(y_true, probs, etiquetas: list[str], destino=None):
    """
    Matriz de confusion. Si se pasa `destino`, guarda el PNG.

    Reemplaza a `modelo_LSA/confusion_matrix.py`, que quedo roto: referencia
    MODEL_NUMS y MODELS_PATH, que ya no existen en constants.py.
    """
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    y_pred = np.argmax(probs, axis=1)
    y_true = np.asarray(y_true)
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(etiquetas)))

    if destino is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 8))
        ConfusionMatrixDisplay(cm, display_labels=etiquetas).plot(
            cmap="Blues", ax=ax, colorbar=False, xticks_rotation=45
        )
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        fig.tight_layout()
        fig.savefig(destino, dpi=150)
        plt.close(fig)

    return cm


def sugerir_epsilon(X, percentil: float = 5.0, max_muestras: int = 1200, seed: int = 42) -> float:
    """
    Elige un eps razonable a partir de los datos.

    La propuesta define eps como "un umbral de vecindad en el espacio de
    entrada" pero no fija su valor, porque depende de la escala de los
    landmarks. Se toma el percentil `percentil` de las distancias entre pares
    distintos: con el default, dos gestos son vecinos si estan entre el 5% de
    los pares mas cercanos del conjunto.

    Es una eleccion documentada y reproducible, que es lo que el informe
    necesita poder justificar.
    """
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    if len(X) > max_muestras:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(len(X), size=max_muestras, replace=False)]

    D = pairwise_distances(X, metric="euclidean")
    fuera_diagonal = D[~np.eye(len(D), dtype=bool)]
    return float(np.percentile(fuera_diagonal, percentil))


def consistency_sigma(X, probs, eps: float) -> dict:
    """
    Consistencia estadistica sigma, tal como se define en la propuesta.

    Devuelve sigma, la cantidad de muestras que tuvieron vecindad util
    (|N_i| > 1) y el tamano promedio de vecindad, que hacen falta para
    interpretar el numero: un sigma calculado sobre 3 muestras no dice nada.
    """
    from sklearn.metrics import pairwise_distances

    X = np.asarray(X, dtype=np.float32).reshape(len(X), -1)
    probs = np.asarray(probs, dtype=np.float64)

    D = pairwise_distances(X, metric="euclidean")
    vecinos = D < eps  # incluye a i consigo mismo, distancia 0

    sigmas = []
    tamanos = []
    for i in range(len(X)):
        idx = np.where(vecinos[i])[0]
        if len(idx) <= 1:
            continue
        # std sesgada (ddof=0) por clase sobre la vecindad, promediada en C.
        desvios = probs[idx].std(axis=0, ddof=0)
        sigmas.append(float(desvios.mean()))
        tamanos.append(len(idx))

    if not sigmas:
        return {
            "sigma": float("nan"),
            "n_con_vecindad": 0,
            "vecindad_media": 0.0,
            "eps": float(eps),
        }

    return {
        "sigma": float(np.mean(sigmas)),
        "n_con_vecindad": len(sigmas),
        "vecindad_media": float(np.mean(tamanos)),
        "eps": float(eps),
    }
