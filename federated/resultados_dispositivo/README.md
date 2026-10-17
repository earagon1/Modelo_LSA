# Entrenamiento de la cabeza en el teléfono (T-F11 / T-F14)

Resultados de `EntrenamientoLocalActivity` (build debug de TraductorLSA): el teléfono lee
las muestras de la sesión guiada de un participante, las lleva al contrato v2 en Kotlin,
saca el embedding con el backbone TFLite congelado y entrena la cabeza (13 261 parámetros)
en Kotlin puro con la tanda 1. Evalúa con la tanda 2, que el modelo exportado no vio
(`federated/exportar_cabeza.py`, `runs/2026-10-16_cabeza`).

## Lenovo TB321FU (Snapdragon 8 Gen 3), P1, 2026-10-17

| Medición | Valor |
|---|---|
| Muestras (entrenamiento / prueba / descartadas) | 48 / 58 / 0 |
| Embedding con el backbone, por muestra | 2,9 ms |
| Una época de la cabeza (48 muestras, lotes de 8), mediana | **20,5 ms** (la primera, 77 ms, incluye el calentamiento) |
| 10 épocas completas | ≈ 0,26 s |
| Memoria extra durante el entrenamiento | ≈ 6 MB |
| Batería | no medible: el entrenamiento dura menos de un segundo |
| Subida por ronda (la cabeza) | 53 KB en float32, 13 KB a 8 bits |
| Accuracy en la tanda 2, antes → después | 82,8 % → 82,8 % |

**Paridad de punta a punta:** la accuracy antes de entrenar (82,8 %) coincide exactamente
con la que da Python sobre las mismas tomas. Contrato v2 en Kotlin + backbone TFLite +
cabeza en Kotlin reproducen el pipeline de entrenamiento. La personalización no cambia
la accuracy de P1 ni en el teléfono ni en Python: el modelo ya la conoce por la webcam y
por la tanda 1. El resultado que vale acá es el costo: entrenar la cabeza en un teléfono
lleva milisegundos.
