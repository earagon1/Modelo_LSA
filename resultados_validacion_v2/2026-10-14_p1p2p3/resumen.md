# Validación v2 (1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109})

| Esquema | n | Accuracy | F1 macro | Por persona |
|---|---|---|---|---|
| publicado | 288 | 0.285 | 0.256 | P1: 0.40, P2: 0.22, P3: 0.24 |
| webcam_a_telefono | 288 | 0.524 | 0.486 | P1: 0.61, P2: 0.49, P3: 0.47 |
| persona | 318 | 0.642 | 0.648 | P1: 0.58, P2: 0.66, P3: 0.68 |
| sesion | 1409 | 0.773 | 0.759 | P1: 0.78, P2: 0.81, P3: 0.69 |
| estratificado | 1409 | 0.941 | 0.923 | P1: 0.97, P2: 0.87, P3: 0.72 |

Modelo publicado: la app muestra el 83 % de las predicciones (confianza ≥ 0.7); de esas, acierta el 32 %.
