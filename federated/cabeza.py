"""
FedAvg de la cabeza con el backbone congelado (PY-F4).

En el teléfono no se entrena la LSTM: on-device con TFLite exige ops Flex y es frágil.
Lo que sí es viable es entrenar las capas densas del final (≈13 k parámetros) sobre
el embedding que da el backbone congelado. Este módulo hace exactamente eso en el
simulador, para que el experimento mida lo mismo que podría correr un celular:

  - el backbone (las dos LSTM) se fija con los pesos del modelo base, entrenado en
    el servidor solo con la webcam de P1;
  - cada cliente entrena localmente solo la cabeza y sube solo la cabeza (o su
    delta comprimido);
  - el servidor promedia con `fedavg.aggregate`, el mismo que ya está testeado.

Lo que viaja por la red es solo lo entrenable: los bytes que se reportan son los de
la cabeza, no los del modelo entero.
"""
from __future__ import annotations

import importlib.util
import os

import numpy as np
import tensorflow as tf
from agregacion import ESTRATEGIAS, combinar
from compresion import comprimir_update, error_relativo
from fedavg import aggregate, reset_optimizer_state

# federated/model.py se carga por ruta y con otro nombre: "model" a secas choca con el
# model.py de la raíz del repo (los dos están en el sys.path de los tests).
_spec = importlib.util.spec_from_file_location(
    "federated_model", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py"))
_modulo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_modulo)
build_model = _modulo.build_model


def construir(pesos_base: list[np.ndarray], t: int, d: int, n_clases: int, lr: float = 1e-3) -> tf.keras.Model:
    """La red de producción con los pesos del modelo base y las LSTM congeladas."""
    modelo = build_model(t, d, n_clases)
    modelo.set_weights(pesos_base)
    for capa in modelo.layers:
        if isinstance(capa, tf.keras.layers.LSTM):
            capa.trainable = False
    # Hay que recompilar después de cambiar `trainable` para que Keras lo respete.
    modelo.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return modelo


def cabeza(modelo: tf.keras.Model) -> list[np.ndarray]:
    return [v.numpy().copy() for v in modelo.trainable_weights]


def poner_cabeza(modelo: tf.keras.Model, pesos: list[np.ndarray]) -> None:
    for v, w in zip(modelo.trainable_weights, pesos, strict=True):
        v.assign(w)


def n_parametros_cabeza(modelo: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in modelo.trainable_weights))


def entrenar_local(modelo, pesos_cabeza, X, y1h, epocas, batch_size=8):
    """Un cliente: arranca de la cabeza global, entrena E épocas y devuelve su cabeza."""
    poner_cabeza(modelo, pesos_cabeza)
    reset_optimizer_state(modelo)
    modelo.fit(X, y1h, epochs=epocas, batch_size=batch_size, shuffle=True, verbose=0)
    return cabeza(modelo)


def fedavg(modelo, clientes, X_test, y_test1h, rondas=40, epocas_locales=2, batch_size=8,
           bits=None, semilla=0, agregacion="fedavg", **parametros_agregacion):
    """
    FedAvg sobre la cabeza. `clientes` es una lista de (X, y_onehot), una por persona.
    Con `bits`, cada cliente sube su delta cuantizado (compresion.py) y se cuentan
    los bytes reales de esa subida.

    `agregacion` = "mediana", "recortada" o "krum" cambia el promedio ponderado por la
    combinación robusta de los deltas (`agregacion.py`, la misma que usa el servidor;
    `parametros_agregacion` son beta, f y m). Con "fedavg" el camino es el de siempre.

    Devuelve la cabeza final y la historia por ronda (accuracy en el test, bytes).
    """
    if agregacion not in ESTRATEGIAS:
        raise ValueError(f"agregación {agregacion!r}: las opciones son {', '.join(ESTRATEGIAS)}")
    rng_q = np.random.default_rng(semilla + 9973)
    global_ = cabeza(modelo)
    bytes_cabeza = int(sum(w.size for w in global_) * 4)
    historia = {"ronda": [], "accuracy": [], "bytes_subida": [], "error_compresion": []}
    subida = 0
    for ronda in range(1, rondas + 1):
        updates, errores = [], []
        for X, y1h in clientes:
            local = entrenar_local(modelo, global_, X, y1h, epocas_locales, batch_size)
            if bits is None:
                updates.append((local, len(X)))
                subida += bytes_cabeza
            else:
                delta = [a - b for a, b in zip(local, global_, strict=True)]
                rec, n_bytes = comprimir_update(delta, bits, rng_q)
                updates.append((rec, len(X)))
                subida += n_bytes
                errores.append(error_relativo(delta, rec))
        if agregacion == "fedavg":
            promedio = aggregate(updates)
            global_ = promedio if bits is None else [g + d for g, d in zip(global_, promedio, strict=True)]
        else:
            # Sin bits, los updates son cabezas: se pasan a deltas para que la combinación
            # robusta vea lo mismo que en el servidor (para la mediana y Krum da igual,
            # pero así hay un solo camino).
            deltas = updates if bits is not None else [
                ([a - b for a, b in zip(local, global_, strict=True)], n) for local, n in updates]
            combinado = combinar(deltas, agregacion, **parametros_agregacion)
            global_ = [(g + c).astype(np.float32) for g, c in zip(global_, combinado, strict=True)]
        poner_cabeza(modelo, global_)
        _, acc = modelo.evaluate(X_test, y_test1h, verbose=0)
        historia["ronda"].append(ronda)
        historia["accuracy"].append(float(acc))
        historia["bytes_subida"].append(int(subida))
        historia["error_compresion"].append(float(np.mean(errores)) if errores else 0.0)
    return global_, historia
