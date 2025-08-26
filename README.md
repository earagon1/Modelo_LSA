Este es un modelo de una red neuronal que traduce Lengua de Señas Argentina(LSA) a texto (y voz). Utilicé MediaPipe para obtener los puntos de la seña y para el entrenamiento usé TensorFlow y Keras.

## SCRIPTS PRINCIPALES
- capture_samples.py → captura las muestras y las ubica en la carpeta frame_actions.
- normalize_samples.py → normaliza las muestras para que todas tengan la misma cantidad de frames (importante).
- create_keypoints.py → crea los keypoints que se usarán en el entrenamiento.
- training_model.py → entrena la red neuronal.
- evaluate_model.py → donde se realiza la prueba de la red neuronal.
- main.py → donde se utiliza una GUI para usar el traductor.

## SCRIPTS SECUNDARIOS
- model.py → aquí se ajusta el modelo de la red neuronal.
- constants.py → ajustes de la red neuronal.
- helpers.py → funciones que se utilizan en los scripts principales.

## Cada vez que quieras mejorar tu modelo:

1. Capturás nuevas muestras con capture_samples.py.
2. Ejecutás normalize_samples.py.
3. Ejecutás create_keypoints.py.
4. Ejecutás training_model.py.
5. Ejecutás convert_to_tflite.py(cambiar el nombre de script )

Todo ese ciclo queda en Python. El teléfono solo necesita el nuevo .tflite.