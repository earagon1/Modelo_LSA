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
Durante las pruebas internas con un conjunto reducido de datos:
- **Accuracy global:** ~90% en validación cruzada.
- **F1-score macro:** ~88% (promedio por clase).
- **Consistencia local:** Alta robustez ante variaciones menores de velocidad y ángulo.

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