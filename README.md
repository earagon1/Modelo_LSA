# Traductor de Lengua de Señas Argentina (LSA) – Entrenamiento del Modelo

Este proyecto forma parte de una tesina que desarrolla un **traductor móvil en tiempo real de Lengua de Señas Argentina (LSA) a texto y voz, y de voz a texto**, completamente funcional **sin conexión a Internet**, ejecutando todo el procesamiento localmente en Android mediante TensorFlow Lite.

En esta parte del repositorio se encuentra el **pipeline de entrenamiento en Python**, que permite:
- Capturar y preparar muestras de datos.
- Entrenar el modelo LSTM con Keras/TensorFlow.
- Exportar el modelo optimizado para dispositivos móviles.

---

## **Resumen técnico**
- **Captura de landmarks:** Se utiliza [MediaPipe](https://developers.google.com/mediapipe) para extraer los **42 puntos clave tridimensionales** de ambas manos, en secuencias de **15 frames** por seña.
- **Entrenamiento:** Se usa una **red LSTM profunda** para clasificar las secuencias de landmarks y convertirlas en palabras.  
- **Conversión:** El modelo entrenado se convierte a TensorFlow Lite (`.tflite`) para ser integrado en la aplicación Android desarrollada en Kotlin.
- **Procesamiento local:** El modelo resultante permite inferencia en tiempo real **sin conexión a internet**, optimizado para ejecutarse en dispositivos móviles de gama media.

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

## **Optimización para TensorFlow Lite**
Para la integración en Android, el modelo se exporta en diferentes variantes:
- **`actions_15_f32.tflite`** → Inferencia en `float32` (máxima precisión).  
- **`actions_15_opt.tflite`** → Cuantización dinámica (pesos `int8`, E/S en `float32`), reduciendo el tamaño del modelo y mejorando la velocidad de inferencia.  

Estas optimizaciones permiten inferencia **en tiempo real**, incluso en dispositivos de gama media.

---

## **Scripts principales**
- **`capture_samples.py`** → Captura muestras y las guarda en `frame_actions/`.
- **`normalize_samples.py`** → Normaliza el número de frames en cada muestra.
- **`create_keypoints.py`** → Genera los keypoints para el entrenamiento.
- **`training_model.py`** → Entrena la red neuronal y guarda el modelo `.keras`.
- **`export_tflite.py`** → Convierte el modelo a formato `.tflite` optimizado.
- **`evaluate_model_tflite.py`** → Permite probar el modelo TFLite antes de integrarlo en Android.
- **`main.py`** → Interfaz gráfica básica para usar el traductor desde PC.

## **Scripts secundarios**
- **`model.py`** → Define la arquitectura de la red neuronal.
- **`constants.py`** → Configuración de rutas y parámetros del modelo.
- **`helpers.py`** → Funciones auxiliares para captura y preprocesamiento.

---

## **Flujo para mejorar el modelo**
1. Capturar nuevas muestras con `capture_samples.py`.
2. Normalizar con `normalize_samples.py`.
3. Generar keypoints con `create_keypoints.py`.
4. Entrenar el modelo con `training_model.py`.
5. Exportar el modelo a `.tflite` con `export_tflite.py`.
6. Probar la inferencia con `evaluate_model_tflite.py`.

---

## **Tecnologías utilizadas**
- Python 3.10+
- TensorFlow / Keras
- MediaPipe
- NumPy, OpenCV
- TensorFlow Lite para exportación

## **Resultados del entrenamiento**
https://youtu.be/lMqT3cidfIA