# Duración real de la captura por toma (auditoría, X4)

Generado el 2026-11-14 con `scripts/duracion_captura.py` (solo lee; tablas completas en `tablas.md`,
números en `resultados.json`). Mediana (mín-máx). La auditoría lo daba como *inferido*: webcam
≈ 0,5 s y teléfono entre 1 y 1,5 s.

## Teléfono: medido

La app guarda en cada frame `t_ms`, los milisegundos desde que la captura pasó a CAPTURING
(`SystemClock.uptimeMillis()` al analizar el frame; `MaquinaDeCaptura.kt`). También guarda por toma
`duracion_ms`, que es el `t_ms` del último frame, y `fps = n_frames / duracion`. Son marcas reales.
Datos: `data/captura_E3/P1..P3/lsa_samples_v2.jsonl`, sesión 1 (bloques A y N) y sesión 2 (B), 449
tomas, todas con mano en todos los frames.

| Dispositivo | Persona | Tomas | Frames por toma | Duración ms | FPS |
|---|---|---|---|---|---|
| Lenovo TB321FU | P1 | 154 | 15 (154) | 876 (782-1331) | 17,1 (11,3-19,2) |
| OnePlus CPH2663 | P2 | 153 | 10 (104), 15 (49) | 1402 (1075-2285) | 7,6 (5,6-14,0) |
| Samsung SM-A556E | P3 | 142 | 10 (131), 15 (11) | 1618 (825-2882) | 6,3 (4,8-12,1) |

- Solo la sesión 1 (A + N, lo que entra al entrenamiento) da casi lo mismo: 874 / 1455 / 1608 ms
  (bloque A). La tabla por bloque está en `tablas.md`.
- Por cantidad de frames: OnePlus 10 frames 1356 ms y 15 frames 1795 ms; Samsung 10 frames 1591 ms
  y 15 frames 2413 ms; tablet 15 frames 876 ms.
- Entre el primer y el último frame guardado: 819 / 1264 / 1452 ms. El primer frame llega un
  intervalo después del inicio.
- Antes de cada captura hay 500 ms de espera con la mano a la vista (WAITING). Esos frames no se
  guardan. La seña puede empezar en esa espera, y eso no queda registrado.
- Son las tomas grabadas con el APK de la captura E3 (octubre). Después (T-A21/22), los FPS
  subieron en los tres equipos, según `plan/progreso.md`. Las capturas de la app actual son más
  cortas, pero eso no sale de estos archivos.

**Respuesta:** «entre 1 y 1,5 s» vale para el OnePlus. La tablet queda por debajo (≈ 0,9 s) y el
Samsung por encima (≈ 1,6 s). La duración depende del dispositivo: el objetivo de frames (10/15/20)
según los FPS no la iguala.

## Webcam: no se puede medir con los datos

- En `data/landmarks/webcam_v2.jsonl` (1200 tomas: 1191 de 15 frames, 6 de 14 y 3 de 13) `t_ms`,
  `fps = 30` y `duracion_ms = 500` **no son mediciones**. `scripts/migrar_v1.py` los escribe
  suponiendo 30 FPS (`t_ms = i·1000/30`).
- Los JPG de `frame_actions/` no tienen marca de tiempo por frame. `capture_samples.py` los guarda
  todos juntos al final de la toma.
- `normalize_samples.py` llevó cada toma a 15 frames: si sobraban, submuestreó y borró; si faltaban,
  interpoló. No dejó anotada la cantidad original. Así, los 15 frames de una toma de webcam pueden
  cubrir cualquier duración. Los respaldos de `D:\Respaldo-Tesis` empiezan el 2026-10-05, después de
  la normalización, así que tampoco ahí están los originales.
- Los FPS reales del bucle de grabación no se registraron. El bucle corre Holistic en CPU y
  `waitKey(10)` en cada frame: probablemente menos de 30 FPS, sin medir.
- Lo único medido es el instante de guardado de cada toma: el nombre `sample_<AAMMDDhhmmssffffff>`,
  `datetime.now()` justo antes de guardar. Entre dos tomas seguidas de la misma clase pasa entera la
  captura de la segunda, así que el intervalo es una **cota superior** de su duración (se descartan
  los intervalos de más de 60 s):

| Año de grabación | Intervalos | Mediana s | Mínimo s | p10 s |
|---|---|---|---|---|
| 2024 | 746 | 3,42 | 1,75 | 2,41 |
| 2025 | 139 | 13,17 | 2,42 | 2,81 |
| 2026 | 218 | 3,86 | 2,15 | 2,58 |

**Se puede afirmar:** cada toma de webcam usada para entrenar tiene 13 a 15 frames después de
normalizar. Las tomas grabadas en tandas rápidas duraron, como máximo, entre 1,75 y 3,4 s (mínimo y
mediana de la cota). **No se puede afirmar:** que cubran 0,5 s, ni ninguna otra duración. Ese
número sale de suponer 30 FPS y 15 frames originales, y ninguna de las dos cosas está medida.
Para saberlo habría que grabar de nuevo con marcas de tiempo, sin normalizar.
