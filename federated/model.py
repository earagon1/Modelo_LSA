"""
Arquitectura del clasificador usada en los experimentos federados.

Es deliberadamente la MISMA red que `modelo_LSA/model.py`, incluidos los
regularizadores L2: si el modelo cambiara entre el experimento centralizado y
el federado, la comparacion no diria nada. Se reescribe aca en lugar de
importarse porque `modelo_LSA/model.py` arrastra `constants.py`, que a su vez
importa cv2 y resuelve rutas contra `os.getcwd()`.

Cualquier cambio en `modelo_LSA/model.py` hay que replicarlo aca. El test
`test_fedavg.py::test_arquitectura_replica_el_original` lo verifica.
"""

from __future__ import annotations

import tensorflow as tf


def build_model(t: int, d: int, n_classes: int) -> tf.keras.Model:
    """LSTM(64) -> LSTM(128) -> Dense(64) -> Dense(64) -> Dense(C, softmax)."""
    modelo = tf.keras.Sequential(
        [
            tf.keras.layers.LSTM(
                64,
                return_sequences=True,
                input_shape=(t, d),
                kernel_regularizer=tf.keras.regularizers.l2(0.01),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.LSTM(
                128,
                return_sequences=False,
                kernel_regularizer=tf.keras.regularizers.l2(0.001),
            ),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(
                64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dense(
                64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.001)
            ),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return modelo


def contar_parametros(modelo: tf.keras.Model) -> int:
    return int(sum(tf.size(w).numpy() for w in modelo.weights))
