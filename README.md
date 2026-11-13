# Traductor de Lengua de Señas Argentina (LSA) – Entrenamiento del Modelo

Este proyecto forma parte de una tesina que desarrolla un **traductor móvil en tiempo real de Lengua de Señas Argentina (LSA) a texto y voz, y de voz a texto**, completamente funcional **sin conexión a Internet**, ejecutando todo el procesamiento localmente en Android mediante TensorFlow Lite.

En este repositorio está el **pipeline en Python**: los datos, el entrenamiento, la validación, el export a TensorFlow Lite, la publicación del modelo en la app y el servidor del aprendizaje federado.

---

## **Resumen técnico**
- **Landmarks:** MediaPipe HandLandmarker, el mismo extractor que usa la app (`lsa/extractor.py`): 21 puntos tridimensionales por mano, las dos manos, en secuencias de **15 frames** por seña.
- **Contrato de features** (`contract/`): la cuenta que convierte los landmarks en la entrada del modelo (`lsa/contrato.py`) es la misma que hace la app. La carpeta está espejada en los dos repositorios y un fixture verifica que Python y Kotlin den el mismo resultado.
- **Modelo:** una red LSTM que clasifica cada secuencia en una de 13 clases (12 señas y «nada»).
- **Export:** TensorFlow Lite con cuantización dinámica y **sin ops Flex**, para que la app no cargue el delegado Flex (`lsa/export.py`).
- **Datos:** 1409 tomas, 1091 de cámara web (una persona) y 318 de teléfonos (tres personas). No están en el repositorio: `data/`, `frame_actions/` y `models/` están en `.gitignore`.

---

## **Arquitectura de la red neuronal**
La arquitectura está diseñada para balancear **precisión y rendimiento** en dispositivos móviles:
- **Entrada:** Secuencias de tamaño `15x126` (15 frames × 42 puntos × 3 coordenadas).  
- **Bloques LSTM:**
  - `LSTM(64, return_sequences=True)` + `Dropout(0.5)`  
  - `LSTM(128, return_sequences=False)` + `Dropout(0.5)`  
- **Capas densas:**
  - `Dense(64, activation='relu')`
  - `Dense(64, activation='relu')`
  - `Dense(C, activation='softmax')` (donde **C = cantidad de clases de señas**).

Este diseño permite capturar dependencias temporales y patrones gestuales, con un modelo liviano para ejecución local.

---

## **Resultados y métricas**
Validación v2 (`resultados_validacion_v2/2026-10-16_p1p2p3_5semillas/`): 1409 tomas de 13 clases
(1091 de cámara web, 318 de teléfonos de tres personas), cinco semillas. Todas las cifras son
sobre las 318 tomas de los teléfonos:

| Esquema | Accuracy | F1 macro |
|---|---|---|
| **Persona no vista (LOPO)**: el uso real de la app | **67,3 %** ± 2,3 entre semillas (60,0-73,0 % entre personas) | 67,3 % |
| Persona conocida, sesión nueva (sin la clase «nada»; `resultados_robustez/`) | 84,6 % | — |
| Tomas mezcladas al azar (estratificado) | 81,4 % | 81,6 % |

- El "~90 %" que figuraba antes salía de una validación vieja, con una sola persona y las tomas
  mezcladas. No describe a una persona nueva.
- El estratificado sobre **todas** las tomas da 94,3 %, porque las de la webcam se reconocen casi sin
  error. La diferencia con LOPO sobre la misma población es de unos 14 puntos, no de 27.
- La consistencia σ, la robustez por condición y la calibración están en `resultados_consistencia/`,
  `resultados_robustez/` y `resultados_calibracion/`.

Estas métricas se calculan con **accuracy, F1-score y matrices de confusión**, y se complementan con un análisis de estabilidad que mide la consistencia de predicciones frente a variaciones mínimas en los datos.

---

## **Flujo actual**
Todo se corre desde la raíz del repositorio. La lógica está en el paquete `lsa/` y los scripts, en `scripts/`.

| Paso | Comando | Qué deja |
|---|---|---|
| Validar | `python scripts/validar.py --etiqueta <nombre>` | `resultados_validacion_v2/<fecha>_<nombre>/`: acierto por esquema (persona no vista, sesión no vista, estratificado) |
| Entrenar el modelo a publicar | `python scripts/entrenar_final.py` | `runs/<fecha>_final/`: `modelo.keras`, `actions_15_opt.tflite`, `words.json` y el fixture de paridad |
| Calibrar | `python scripts/calibrar.py` | `resultados_calibracion/`: temperatura y umbral de confianza |
| Publicar en la app | `python scripts/publicar_modelo.py --desde runs/<fecha>_final --calibracion resultados_calibracion/calibracion_app.json` | Los cuatro assets de la app, con su manifiesto `modelo.json` |
| Revisar lo publicado | `python scripts/publicar_modelo.py --verificar` | — |

`publicar_modelo.py` es el único camino para escribir los assets de la app: comprueba que el modelo, las etiquetas y la calibración correspondan entre sí antes de copiarlos.

Otros análisis: `scripts/robustez.py` (condiciones adversas), `scripts/consistencia.py` (métrica σ) y `scripts/reporte_tesis.py`, que junta todo en `resultados_tesis/` para el informe.

Tests: `python -m pytest` (la suite pasa en un clon limpio, sin los datos).

---

## **Aprendizaje federado**
`federated/` tiene dos cosas: los experimentos de FedAvg en simulación y el servidor que corre en una PC de la red local y coordina rondas con la app. Ver `federated/README.md`.

---

## **Scripts heredados (no usar para entrenar)**
En la raíz quedan los scripts del primer pipeline, que trabajaba solo con la cámara web. Se conservan porque generaron los datos originales, pero **varios pisan o borran datos**:

- `normalize_samples.py` borra frames de las tomas.
- `create_keypoints.py` sobrescribe todos los `.h5`.
- `training_model.py` y `export_tflite.py` sobrescriben `models/`, y el primero reescribe `words.json`.
- `finetune_from_lsa_json.py` arma las etiquetas en orden alfabético y cruza las clases.

`capture_samples.py` sigue sirviendo para grabar con la cámara web, y `model.py` define la arquitectura que usan los scripts nuevos. El detalle de qué no correr está en `CLAUDE.md`.

---

## **Tecnologías utilizadas**
- Python 3.10 (entorno fijado en `requirements.txt`)
- TensorFlow 2.10 / Keras
- MediaPipe
- NumPy, scikit-learn, OpenCV

## **Resultados del entrenamiento**
https://youtu.be/lMqT3cidfIA
