"""Export a TFLite sin ops Flex (PY-X1).

`export_tflite.py` convierte con `_experimental_lower_tensor_list_ops=False`, lo que
deja las LSTM como ops Flex (`FlexTensorListReserve/SetItem/Stack`). Eso obliga a la
app a cargar `tensorflow-lite-select-tf-ops` (decenas de MB por ABI) y hace que el
int8 caiga en Flex y sea 20 veces más lento.

Acá la conversión parte de una función concreta con forma fija [1, T, D] y de la
red **desenrollada** (`LSTM(unroll=True)`, con los mismos pesos): con T=15 las LSTM
quedan como ops builtin comunes (MatMul, Add, Logistic, Tanh…), sin TensorList.

Por qué no la LSTM fusionada: con forma fija, TF 2.10 la convierte en el op builtin
`UNIDIRECTIONAL_SEQUENCE_LSTM`, pero el resultado está **mal**. Medido el 2026-10-05
con `models/actions_15.keras` sobre las 1200 secuencias, su top-1 coincide con
Keras en el 51 % (también en CPU). La versión desenrollada coincide en el 100 %
(f32: error máximo 4e-7). Con pesos aleatorios el error de la fusionada casi no se
ve (0,003), así que una prueba solo con modelo aleatorio no alcanza: siempre hay que
validar contra el modelo entrenado (`scripts/probar_export_sin_flex.py --modelo`).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import tensorflow as tf


def desenrollar(modelo: tf.keras.Model) -> tf.keras.Model:
    """Copia de un Sequential con todas sus LSTM en `unroll=True` y los mismos pesos.

    Desenrollar no cambia la matemática: Keras da salidas idénticas bit a bit
    (verificado), solo cambia el grafo que ve el conversor.
    """
    config = modelo.get_config()
    for capa in config["layers"]:
        if capa["class_name"] == "LSTM":
            capa["config"]["unroll"] = True
    copia = tf.keras.Sequential.from_config(config)
    copia.build(modelo.input_shape)
    copia.set_weights(modelo.get_weights())
    return copia


def _funcion_concreta(modelo: tf.keras.Model, t: int, d: int):
    @tf.function(input_signature=[tf.TensorSpec([1, t, d], tf.float32, name="secuencia")])
    def inferir(x):
        return modelo(x, training=False)
    return inferir.get_concrete_function()


def convertir(modelo: tf.keras.Model, t: int, d: int, variante: str = "f32",
              dataset_representativo: Callable[[], Iterable] | None = None) -> bytes:
    """Convierte a TFLite solo con ops builtin. Recibe el modelo tal cual (desenrolla
    adentro).

    variante:
      - "f32": sin cuantizar.
      - "dinamica": pesos int8, activaciones float (Optimize.DEFAULT).
      - "int8": cuantización entera con calibración; entrada y salida quedan en
        float32 para no tocar TFLiteClassifier.kt.
    Nunca habilita SELECT_TF_OPS: si la red necesita Flex, la conversión falla
    acá y no en el teléfono.
    """
    modelo = desenrollar(modelo)
    conv = tf.lite.TFLiteConverter.from_concrete_functions([_funcion_concreta(modelo, t, d)], modelo)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    if variante in ("dinamica", "int8"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if variante == "int8":
        if dataset_representativo is None:
            raise ValueError("int8 necesita dataset_representativo")
        conv.representative_dataset = dataset_representativo
        conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
    return conv.convert()


def ops_del_modelo(tflite: bytes) -> list[str]:
    """Nombres de los ops del grafo TFLite, en orden. Sirve para afirmar que no hay Flex."""
    interp = tf.lite.Interpreter(model_content=tflite)
    return [op["op_name"] for op in interp._get_ops_details()]


def ops_flex(tflite: bytes) -> list[str]:
    """Ops custom o Flex presentes; tiene que dar lista vacía."""
    return sorted({o for o in ops_del_modelo(tflite) if o.startswith("Flex") or o == "CUSTOM"})


def inferir_tflite(tflite: bytes, x: np.ndarray) -> np.ndarray:
    """Corre el .tflite sobre un lote, una secuencia por vez (el modelo tiene batch 1)."""
    interp = tf.lite.Interpreter(model_content=tflite)
    interp.allocate_tensors()
    ent, sal = interp.get_input_details()[0], interp.get_output_details()[0]
    out = []
    for seq in np.asarray(x, dtype=np.float32):
        interp.set_tensor(ent["index"], seq[None])
        interp.invoke()
        out.append(interp.get_tensor(sal["index"])[0])
    return np.stack(out)
