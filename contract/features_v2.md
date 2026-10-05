# Contrato de features v2 (J-01)

Define **qué entra al modelo** de SeÑAR, de forma idéntica en Python (entrenamiento,
evaluación, calibración) y en la app Android (inferencia y dataset del teléfono).
La versión legible es este archivo. Los parámetros que leen los programas están en
`features_v2.json`, y la prueba de que las dos implementaciones coinciden está en
`fixtures/paridad_features_v2.json`.

La carpeta `contract/` está **espejada** en `modelo_LSA/contract/` y en
`TraductorLSA/contract/`. `SHA256SUMS` fija el contenido, y cada repo tiene un test
que falla si su copia no coincide. La implementación de referencia es
`modelo_LSA/lsa/contrato.py`, y los fixtures se regeneran con
`scripts/generar_fixtures_paridad.py`.

## Por qué existe

Hasta la v1, la webcam y el teléfono producían vectores distintos para la misma seña
(`plan/modelo_lsa.md` §1.4). Lo que se midió con los datos reales el 2026-10-05
(`plan/progreso.md`):

- **Lateralidad:** HandLandmarker Tasks, sobre frames **sin espejar**, etiqueta la mano
  del sujeto igual que Holistic. El mapeo `"Left"→lh`, `"Right"→rh` que ya usa la app
  coincide con el entrenamiento. *La sospecha de manos cruzadas (H-A03) no se confirmó
  en la PC; falta confirmarla en el dispositivo.*
- **Frames vacíos:** el 32,8 % de los frames de entrenamiento no tiene manos (hasta el
  71 % en «por favor»), mientras que la app nunca le pasa al modelo un frame sin mano.
- **Truncado:** Python se quedaba con los primeros 15 frames y la app remuestreaba
  10/15/20. En la webcam, eso dejó la entrada de las manos y cortó el final de la seña.
- **Aspecto:** la app lleva x,y a un lienzo cuadrado y Python no.

## Reglas

### 1. Extractor
MediaPipe **HandLandmarker Tasks**, con el archivo `hand_landmarker.task` cuyo sha256
está en `features_v2.json`. Es el mismo archivo en la PC y en el teléfono.
- `num_hands = 2`.
- Umbrales por defecto: detección 0,6; presencia y tracking 0,5. Son los de la app con
  sensibilidad 0,6: presencia = tracking = sensibilidad − 0,1.

### 2. Imagen de entrada
- El frame va **sin espejar** (tal como lo entrega la cámara o la webcam) y **rotado a
  "arriba"** antes de detectar: 0, 90, 180 o 270 grados.
- En el modo seña, la app queda **fija en vertical** (decisión D-7).
- `W` y `H` son el ancho y el alto **de la imagen ya rotada**.
- La cámara (frontal o trasera) **no cambia** las features: las dos entregan frames
  sin espejar.

### 3. Lateralidad
Cada mano detectada trae su etiqueta de HandLandmarker (la categoría de mayor score).
- `"Left"` → slot `lh`, `"Right"` → slot `rh`, cualquier otra → `rh`.
- **Etiqueta repetida** (las dos manos con la misma etiqueta): la de mayor score se
  queda con su slot y la otra va al slot libre. Si los scores empatan, gana la que
  vino primero en el resultado.
- Se usan como máximo dos manos, en el orden en que llegan.

### 4. Coordenadas: corrección a lienzo cuadrado
Con `M = max(W, H)`, para cada landmark:
```
x' = x · (W/M) + (1 − W/M)/2
y' = y · (H/M) + (1 − H/M)/2
z' = z
```
Las cuentas son en float32, en el mismo orden que la app: `sx = W/M`, `ox = (1 − sx)/2`
y `x' = x·sx + ox`.

### 5. Vector por frame
126 valores float32: `lh[21×(x,y,z)] ‖ rh[21×(x,y,z)]`, en el orden de los landmarks
de MediaPipe (0 = muñeca … 20), con ceros en el slot de la mano que falta.

### 6. Frames sin mano
Un frame **sin ninguna mano detectada no forma parte de la secuencia**: se descarta
antes de remuestrear. Esto ya lo hace la app al capturar, y en Python se aplica al
cargar o re-extraer cualquier fuente, también la v1.

Para los datos v1 se descartan **todos** los frames vacíos, no solo los de los bordes,
porque es la única forma de que coincida con lo que ve la app. Si una toma queda con
menos de `min_frames_validos` frames, se excluye y se reporta.

### 7. Secuencia temporal
- Se guardan los **frames crudos**, sin remuestrear, en el dataset v2 del teléfono y en
  `landmarks_v2` de la PC.
- La captura no depende de los FPS: es **por tiempo** (`captura.duracion_ms`). El valor
  es provisional y se calibra con la sesión de captura (E3).
- El remuestreo a `T = 15` es determinista, con `n` = frames válidos:
  - `n == T`: sin cambios.
  - `n < T`: interpolación lineal en `linspace(0, n−1, T)`. Para cada `v`: `li = floor(v)`,
    `ui = ceil(v)`, `w = float32(v − li)`, y la salida es `(1 − w)·a[li] + w·a[ui]`, en
    float32. Si `n == 1`, se repite el frame.
  - `n > T`: submuestreo `idx_i = floor(i·n / T)`, calculado **con enteros**:
    `(i*n) div T`. Nunca con un paso en coma flotante acumulado.
  - `n == 0`: no hay predicción ni muestra.
- **Nunca** se trunca a los primeros frames ni se rellena con ceros.

### 8. Versión
`"features_version": "2"` va en cada muestra, en cada archivo de landmarks y en el
manifiesto del modelo. La app rechaza un modelo con otra versión, y Python no mezcla
v1 con v2 sin una marca explícita.

## Paridad (J-02)
`fixtures/paridad_features_v2.json` trae casos con landmarks crudos de entrada (N =
8/10/15/20/27 frames × vertical/horizontal × frontal/trasera, más casos borde:
etiqueta repetida, mano que falta, frames vacíos, N = 1) y la matriz 15×126 esperada.
- `tests/test_contrato.py` (Python) y el unit test de Kotlin (T-A11) tienen que
  reproducirla con una tolerancia de `1e-6`.
- Un cambio en las reglas implica: subir la versión, regenerar los fixtures, actualizar
  `SHA256SUMS` y copiar `contract/` al otro repo.
