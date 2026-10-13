# Sesión de captura E3: comparativa por participante y dispositivo

Generado por `scripts/resumen_captura.py` a partir de `data/captura_E3/*/lsa_samples_v2.jsonl`.
No editar a mano.

| | P1 | P2 |
|---|---|---|
| Dispositivo | LENOVO TB321FU | OnePlus CPH2663 |
| Android | 16 | 16 |
| Tomas | 106 | 106 |
| FPS (mediana) | 17,1 | 7,7 |
| FPS (mín–máx) | 11,3–18,0 | 6,1–14,0 |
| Frames por toma | 15: 106 | 10: 62, 15: 44 |
| Duración de una captura (ms, mediana) | 876 | 1418 |
| Duración de la sesión (min) | 9,8 | 19,4 |

Lectura: la app captura una cantidad fija de frames según los FPS (10, 15 o 20), así que
la ventana de tiempo que cubre una toma depende del dispositivo. En un equipo rápido 15 frames
pueden durar menos de un segundo, y una seña larga no entra completa; en uno lento la captura
baja a 10 frames. Es el argumento empírico para que el contrato v2 capture por tiempo.

## % de frames con las dos manos detectadas, por seña

| Seña | P1 | P2 |
|---|---|---|
| adios | 0 % | 0 % |
| bien | 0 % | 0 % |
| como estas | 52 % | 24 % |
| e | 0 % | 0 % |
| gracias | 0 % | 1 % |
| hola | 0 % | 0 % |
| i | 0 % | 0 % |
| l | 0 % | 0 % |
| mi nombre es | 0 % | 0 % |
| n | 0 % | 0 % |
| nada | 35 % | 21 % |
| por favor | 78 % | 55 % |
| v | 0 % | 0 % |

## Versión LaTeX (booktabs)

```latex
\begin{tabular}{lrr}
\toprule
 & P1 & P2 \\
\midrule
Dispositivo & LENOVO TB321FU & OnePlus CPH2663 \\
Android & 16 & 16 \\
Tomas & 106 & 106 \\
FPS (mediana) & 17,1 & 7,7 \\
FPS (mín–máx) & 11,3–18,0 & 6,1–14,0 \\
Frames por toma & 15: 106 & 10: 62, 15: 44 \\
Duración de una captura (ms, mediana) & 876 & 1418 \\
Duración de la sesión (min) & 9,8 & 19,4 \\
\bottomrule
\end{tabular}
```
