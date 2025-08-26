# export_tflite.py (Opción A - Select TF Ops)
# Genera modelos .tflite desde tu .keras habilitando SELECT_TF_OPS
# - Float32 puro (entrada/salida float32)
# - Optimizado con Dynamic Range Quantization (pesos int8, E/S float32)
# - (Opcional) INT8 puro con representative dataset

import os
import numpy as np
import tensorflow as tf

try:
    from keras.models import load_model  # Keras 2 / standalone
except Exception:  # fallback si fuera necesario
    from tensorflow.keras.models import load_model

from constants import (
    MODEL_PATH, MODEL_FOLDER_PATH, MODEL_FRAMES, LENGTH_KEYPOINTS,
    WORDS_JSON_PATH,
)
from helpers import get_sequences_and_labels, get_word_ids


# --- Normalización temporal coherente con evaluate_model/main (T=15 por defecto) ---
def _interpolate_keypoints(seq, target_len=15):
    cur = len(seq)
    if cur == target_len:
        return seq
    idx = np.linspace(0, cur - 1, target_len)
    out = []
    for i in idx:
        li, ui = int(np.floor(i)), int(np.ceil(i))
        w = i - li
        if li == ui:
            out.append(seq[li])
        else:
            out.append((1 - w) * np.array(seq[li]) + w * np.array(seq[ui]))
    return [np.asarray(x) for x in out]


def _normalize_to_fixed(seq, target_len=15):
    cur = len(seq)
    if cur < target_len:
        return _interpolate_keypoints(seq, target_len)
    elif cur > target_len:
        step = cur / target_len
        idx = np.arange(0, cur, step).astype(int)[:target_len]
        return [seq[i] for i in idx]
    else:
        return seq


# --- Representative dataset para cuantización INT8 pura ---
def representative_dataset_gen(max_samples=300):
    words = get_word_ids(WORDS_JSON_PATH)
    sequences, _ = get_sequences_and_labels(words)  # lista de secuencias (cada una: [frames de 126])
    count = 0
    for seq in sequences:
        s = _normalize_to_fixed(seq, int(MODEL_FRAMES))
        arr = np.array(s, dtype=np.float32)[None, ...]  # [1, T, 126]
        yield [arr]
        count += 1
        if count >= max_samples:
            break


# --- Helper para setear Select TF Ops en el conversor ---
def _enable_select_tf_ops(converter: tf.lite.TFLiteConverter):
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Evita el intento de "lowering" de TensorList, que está fallando
    converter._experimental_lower_tensor_list_ops = False  # noqa: SLF001 (atributo privado aceptado por TF)



def main():
    os.makedirs(MODEL_FOLDER_PATH, exist_ok=True)

    # 1) Cargar el modelo .keras entrenado
    print(f"Cargando modelo desde: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    # 2A) Export TFLite float32 (E/S float32) con Select TF Ops
    print("Convirtiendo a TFLite (float32 + Select TF Ops)...")
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    _enable_select_tf_ops(conv)
    tflite_f32 = conv.convert()
    out_f32 = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}_f32.tflite")
    with open(out_f32, "wb") as f:
        f.write(tflite_f32)
    print("Escribí:", out_f32)

    # 2B) Export TFLite optimizado (Dynamic Range Quantization): pesos int8, E/S float32
    print("Convirtiendo a TFLite (opt + Select TF Ops)...")
    conv_opt = tf.lite.TFLiteConverter.from_keras_model(model)
    conv_opt.optimizations = [tf.lite.Optimize.DEFAULT]
    _enable_select_tf_ops(conv_opt)
    tflite_opt = conv_opt.convert()
    out_opt = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}_opt.tflite")
    with open(out_opt, "wb") as f:
        f.write(tflite_opt)
    print("Escribí:", out_opt)

    # 2C) (Opcional) INT8 puro (E/S int8). Requiere representative dataset.
    make_int8 = False
    if make_int8:
        print("Convirtiendo a TFLite (INT8 puro + Select TF Ops)...")
        conv_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
        conv_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        conv_int8.representative_dataset = representative_dataset_gen
        # Forzar INT8 para kernels TFLite; incluimos Select TF Ops por compatibilidad grafo
        conv_int8.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        conv_int8.inference_input_type = tf.int8
        conv_int8.inference_output_type = tf.int8
        conv_int8._experimental_lower_tensor_list_ops = False

        tflite_int8 = conv_int8.convert()
        out_int8 = os.path.join(MODEL_FOLDER_PATH, f"actions_{MODEL_FRAMES}_int8.tflite")
        with open(out_int8, "wb") as f:
            f.write(tflite_int8)
        print("Escribí:", out_int8)

    print("Listo ✅")


if __name__ == "__main__":
    main()
