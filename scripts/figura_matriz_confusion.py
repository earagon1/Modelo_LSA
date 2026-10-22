"""Figura de la matriz de confusión para el informe (persona no vista, cinco semillas).

Lee la matriz acumulada del esquema `persona` de una corrida de `validar.py` y la dibuja
normalizada por fila (qué porcentaje de las tomas de cada seña se predijo como cada
clase). Un solo tono, de claro a oscuro: la magnitud es lo único que codifica el color.
Las celdas en cero quedan vacías para que se lean las confusiones.

Escribe resultados_tesis/matriz_confusion_persona.{pdf,png}.

Uso, desde la raíz de modelo_LSA:
    python scripts/figura_matriz_confusion.py resultados_validacion_v2/2026-10-16_p1p2p3_5semillas
"""
import json
import os
import sys

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOMBRES = {"hola": "hola", "adios": "adiós", "bien": "bien", "como estas": "¿cómo estás?", "gracias": "gracias",
           "e": "E", "v": "V", "l": "L", "i": "I", "n": "N", "mi nombre es": "mi nombre es",
           "por favor": "por favor", "nada": "nada"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    corrida = sys.argv[1]
    with open(os.path.join(corrida, "resultados.json"), encoding="utf-8") as f:
        res = json.load(f)
    m = np.asarray(res["esquemas"]["persona"]["global"]["matriz"], float)
    clases = [NOMBRES.get(c, c) for c in res["clases"]]
    pct = 100 * m / m.sum(axis=1, keepdims=True)

    # Secuencial de un solo tono (azul), del casi blanco al azul oscuro.
    mapa = LinearSegmentedColormap.from_list("azul", ["#f4f8fd", "#9cc3e8", "#2f6fb2", "#0b2f5e"])
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(pct, cmap=mapa, vmin=0, vmax=100)
    for i in range(len(clases)):
        for j in range(len(clases)):
            v = pct[i, j]
            if m[i, j] == 0:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v >= 50 else "#1f2328")
    ax.set_xticks(range(len(clases)), clases, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(clases)), clases, fontsize=8.5)
    ax.set_xlabel("Seña predicha", fontsize=9.5)
    ax.set_ylabel("Seña real", fontsize=9.5)
    for borde in ax.spines.values():
        borde.set_visible(False)
    ax.tick_params(length=0)
    barra = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    barra.set_label("Porcentaje de las tomas de la seña real", fontsize=8.5)
    barra.ax.tick_params(labelsize=8)
    barra.outline.set_visible(False)
    fig.tight_layout()
    salida = os.path.join(RAIZ, "resultados_tesis")
    os.makedirs(salida, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(salida, f"matriz_confusion_persona.{ext}"), dpi=200, bbox_inches="tight")
    n = int(m.sum())
    print(f"{n} predicciones ({len(res['semillas'])} semillas); diagonal media {np.mean(np.diag(pct)):.1f} %")
    print(f"-> {salida}/matriz_confusion_persona.pdf")


if __name__ == "__main__":
    main()
