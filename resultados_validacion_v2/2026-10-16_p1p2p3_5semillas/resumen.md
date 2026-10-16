# Validación v2 (1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109})

Semillas: [0, 1, 2, 3, 4]

| Esquema | n | Accuracy (media ± desvío) | F1 macro (media ± desvío) | Por persona |
|---|---|---|---|---|
| publicado | 288 | 0.285 ± 0.000 | 0.256 ± 0.000 | P1: 0.40, P2: 0.22, P3: 0.24 |
| webcam_a_telefono | 1440 | 0.560 ± 0.054 | 0.533 ± 0.053 | P1: 0.70, P2: 0.49, P3: 0.49 |
| persona | 1590 | 0.673 ± 0.023 | 0.673 ± 0.023 | P1: 0.60, P2: 0.73, P3: 0.69 |
| sesion | 7045 | 0.780 ± 0.071 | 0.751 ± 0.079 | P1: 0.79, P2: 0.73, P3: 0.73 |
| estratificado | 7045 | 0.943 ± 0.004 | 0.921 ± 0.005 | P1: 0.97, P2: 0.84, P3: 0.75 |

Modelo publicado: la app muestra el 83 % de las predicciones (confianza ≥ 0.7); de esas, acierta el 32 %.
