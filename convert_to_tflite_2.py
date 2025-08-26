#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import json
from glob import glob

import numpy as np
import tensorflow as tf

# --------------------------------------------
# Utilidades de IO
# --------------------------------------------

def load_keras_model(model_path: str):
    """
    Carga un modelo Keras desde:
      - Carpeta SavedModel (contiene saved_model.pb y variables/)
      - Archivo .h5/.keras
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No existe: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        raise RuntimeError(
            "Fallo al cargar el modelo. Si usaste custom_objects (layers/metrics/loss), "
            "pásalos con --custom-objects o exportá un SavedModel estándar.\n"
            f"Error original: {e}"
        )

def load_word_ids(words_json_path: str):
    if not words_json_path:
        return None
    with open(words_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("words.json debe ser una lista (array JSON).")
    return data

def guess_input_shape(model: tf.keras.Model):
    """
    Intenta deducir la forma del input (sin batch) desde el modelo.
    Devuelve (frames, features) si posible; de lo contrario None.
    """
    try:
        # Keras moderno: model.inputs[0].shape => (None, T, D)
        shape = model.inputs[0].shape
        # shape: (None, T, D) o (None, T) - según arquitectura
        if len(shape) == 3:
            frames = shape[1]
            features = shape[2]
            if frames is not None and features is not None:
                return int(frames), int(features)
        elif len(shape) == 2:
            # (None, D) -> no temporal
            return None
    except Exception:
        pass
    return None

# --------------------------------------------
# NUEVO: clonado con UNROLL en LSTM/GRU (incluye Bidirectional)
# --------------------------------------------

def _clone_lstm_or_gru(layer):
    """
    Si la capa es LSTM/GRU, fuerza unroll=True preservando el resto de la config.
    Si la capa es Bidirectional(<LSTM/GRU>), reconstruye el wrapper con la base unrolled.
    Si no aplica, devuelve None (para que clone_function mantenga la capa original).
    """
    L = tf.keras.layers
    # LSTM
    if isinstance(layer, L.LSTM):
        cfg = layer.get_config()
        cfg["unroll"] = True
        return L.LSTM.from_config(cfg)
    # GRU (por si acaso)
    if isinstance(layer, L.GRU):
        cfg = layer.get_config()
        cfg["unroll"] = True
        return L.GRU.from_config(cfg)
    # Bidirectional(LSTM/GRU)
    if isinstance(layer, L.Bidirectional):
        base = layer.layer
        if isinstance(base, (L.LSTM, L.GRU)):
            base_cfg = base.get_config()
            base_cfg["unroll"] = True
            new_base = type(base).from_config(base_cfg)
            # recrea el wrapper manteniendo el merge_mode
            return L.Bidirectional(new_base, merge_mode=layer.merge_mode)
    return None

def clone_with_unroll(model: tf.keras.Model) -> tf.keras.Model:
    """
    Clona el modelo aplicando unroll=True en las capas recurrentes.
    Copia los pesos desde el original. No reentrena.
    """
    def edit(layer):
        new_layer = _clone_lstm_or_gru(layer)
        return new_layer if new_layer is not None else layer

    cloned = tf.keras.models.clone_model(model, clone_function=edit)
    cloned.set_weights(model.get_weights())
    return cloned

# --------------------------------------------
# Representative dataset para INT8
# --------------------------------------------

def representative_ds_from_npy_like(path: str, frames: int, features: int, batch_size: int = 1):
    """
    Genera un representative dataset para calibración INT8.
    Soporta:
     - Archivo .npy con shape [N, frames, features]
     - Archivo .npz con clave 'X' o 'arr_0' con shape [N, frames, features]
     - Carpeta con múltiples .npy (cada uno [frames, features] o [N, frames, features])
    """
    arrays = []

    if os.path.isdir(path):
        files = sorted(glob(os.path.join(path, "*.npy")))
        if not files:
            raise FileNotFoundError(f"No hay .npy en {path}")
        for fp in files:
            arr = np.load(fp, allow_pickle=False)
            if arr.ndim == 2 and arr.shape == (frames, features):
                arr = arr[None, ...]  # -> [1, T, D]
            elif arr.ndim == 3 and arr.shape[1:] == (frames, features):
                pass
            else:
                raise ValueError(f"{fp}: forma inesperada {arr.shape}, esperado [T,D] o [N,T,D]")
            arrays.append(arr)
        data = np.concatenate(arrays, axis=0)

    elif path.endswith(".npy"):
        data = np.load(path, allow_pickle=False)
        if data.ndim != 3 or data.shape[1:] != (frames, features):
            raise ValueError(f"{path}: forma inesperada {data.shape}, esperado [N,{frames},{features}]")
    elif path.endswith(".npz"):
        z = np.load(path)
        key = "X" if "X" in z.files else z.files[0]
        data = z[key]
        if data.ndim != 3 or data.shape[1:] != (frames, features):
            raise ValueError(f"{path}:{key}: forma inesperada {data.shape}, esperado [N,{frames},{features}]")
    else:
        raise ValueError("representative path debe ser carpeta con .npy o archivo .npy/.npz")

    # Normalización si tu runtime usa otra escala (aquí asumimos ya preprocesado)
    data = data.astype(np.float32)

    def gen():
        for i in range(data.shape[0]):
            x = data[i]
            # batch_size=1: [1, T, D]
            x = np.expand_dims(x, axis=0).astype(np.float32)
            yield [x]
    return gen

def representative_ds_random(frames: int, features: int, n: int = 200):
    """
    Fallback: dataset sintético para calibrar INT8 si no hay datos reales.
    Recomendación: USAR datos reales del dominio para mejor precisión.
    """
    def gen():
        rng = np.random.default_rng(123)
        for _ in range(n):
            x = rng.normal(0.0, 1.0, size=(1, frames, features)).astype(np.float32)
            yield [x]
    return gen

# --------------------------------------------
# Conversión
# --------------------------------------------

def convert_to_tflite(
    model: tf.keras.Model,
    out_path: str,
    quant: str = "float32",
    frames: int = 15,
    features: int = 126,
    rep_path: str = None,
    use_select_tf_ops: bool = False,   # NUEVO: plan B opcional
):
    """
    Convierte un modelo Keras a TFLite.
      quant: 'float32' | 'float16' | 'int8'
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimizaciones comunes
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # (Opcional) permitir SELECT_TF_OPS + evitar lowering de TensorList
    if use_select_tf_ops:
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        try:
            converter._experimental_lower_tensor_list_ops = False
        except Exception:
            pass

    # Set de cuantización
    quant = quant.lower()
    if quant == "float32":
        # Sin cuantizar (pero mantiene Optimize.DEFAULT para posibles fusiones)
        pass

    elif quant == "float16":
        converter.target_spec.supported_types = [tf.float16]

    elif quant == "int8":
        # Requiere representative dataset
        if rep_path:
            converter.representative_dataset = representative_ds_from_npy_like(
                rep_path, frames=frames, features=features
            )
        else:
            # Fallback (no ideal, pero útil para prototipado)
            converter.representative_dataset = representative_ds_random(
                frames=frames, features=features, n=256
            )

        # Cuantización full-int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # Entrada/Salida int8 (para máxima compatibilidad con NNAPI delegados)
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

    else:
        raise ValueError("quant debe ser float32 | float16 | int8")

    tflite_model = converter.convert()

    # Guarda archivo .tflite
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"[OK] Guardado: {out_path} ({quant})")

    return tflite_model

# --------------------------------------------
# Verificación (Keras vs TFLite)
# --------------------------------------------

def run_tflite_inference(tflite_bytes: bytes, x: np.ndarray):
    """
    Ejecuta un forward TFLite en memoria.
    - x: [N, T, D] float32
    Devuelve: logits [N, C] en float32 (reconvertidos si int8).
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Preparo input teniendo en cuenta tipo del intérprete
    in_dtype = input_details[0]["dtype"]
    x_in = x
    if in_dtype == np.int8:
        # Re-escala a int8 usando quant params del tensor
        scale, zero_point = input_details[0]["quantization"]
        if scale == 0:
            raise ValueError("Escala de cuantización de entrada = 0")
        x_in = np.round(x / scale + zero_point).astype(np.int8)

    # Ejecutar batch por batch (seguro)
    outputs = []
    for i in range(x.shape[0]):
        xi = x_in[i : i + 1]
        interpreter.set_tensor(input_details[0]["index"], xi)
        interpreter.invoke()
        yo = interpreter.get_tensor(output_details[0]["index"])
        # Si salida es int8, dequantize a float32 para comparar
        if yo.dtype == np.int8:
            scale, zero_point = output_details[0]["quantization"]
            yo = (yo.astype(np.float32) - zero_point) * scale
        outputs.append(yo)
    return np.concatenate(outputs, axis=0)

def verify_equivalence(model, tflite_bytes, frames=15, features=126, n=16):
    """
    Compara Keras vs TFLite sobre n muestras aleatorias normales (o
    pasá un set real si querés).
    """
    rng = np.random.default_rng(42)
    x = rng.normal(0.0, 1.0, size=(n, frames, features)).astype(np.float32)

    # Predicción Keras
    y_keras = model.predict(x, verbose=0)
    if isinstance(y_keras, (list, tuple)):
        y_keras = y_keras[0]
    y_tfl = run_tflite_inference(tflite_bytes, x)

    max_abs_diff = np.max(np.abs(y_keras - y_tfl))
    mean_abs_diff = np.mean(np.abs(y_keras - y_tfl))
    argmax_match = np.mean(np.argmax(y_keras, axis=1) == np.argmax(y_tfl, axis=1))

    print(f"[Check] max|Δ| = {max_abs_diff:.6f} | mean|Δ| = {mean_abs_diff:.6f} | argmax match = {argmax_match*100:.1f}%")
    return max_abs_diff, mean_abs_diff, argmax_match

# --------------------------------------------
# CLI
# --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convertir modelo Keras a TFLite (float32/float16/int8) con opción de UNROLL.")
    parser.add_argument("--model", required=True, help="Ruta a SavedModel (carpeta) o .h5/.keras")
    parser.add_argument("--out", required=True, help="Ruta de salida .tflite (p.ej., ./out/model.tflite)")
    parser.add_argument("--quant", default="float32", choices=["float32", "float16", "int8"], help="Esquema de cuantización")
    parser.add_argument("--rep", default=None, help="Ruta a representative dataset (carpeta con .npy, o .npy/.npz). Requerido/ideal para int8.")
    parser.add_argument("--frames", type=int, default=15, help="Frames temporales esperados (default: 15)")
    parser.add_argument("--features", type=int, default=126, help="Features por frame (default: 126)")
    parser.add_argument("--verify", action="store_true", help="Verificar salidas Keras vs TFLite")
    parser.add_argument("--words", default=None, help="Opcional: words.json para validar dimensión de salida")
    parser.add_argument("--unroll", action="store_true", help="Forzar unroll=True en LSTM/GRU (incluye Bidirectional).")
    parser.add_argument("--select-tf-ops", action="store_true", help="(Opcional) Usar SELECT_TF_OPS y desactivar lowering de TensorList (plan B).")
    args = parser.parse_args()

    # Seguridad: fuerza float32 por defecto (evita mixed precision en conversión)
    try:
        tf.keras.mixed_precision.set_global_policy("float32")
    except Exception:
        pass

    # Carga modelo
    model = load_keras_model(args.model)

    # NUEVO: aplicar UNROLL si lo pediste
    if args.unroll:
        print("[INFO] Clonando modelo con unroll=True en LSTM/GRU (incluye Bidirectional)...")
        model = clone_with_unroll(model)

    # Verifica forma de entrada (opcional pero útil)
    guessed = guess_input_shape(model)
    if guessed is not None:
        gt, gd = guessed
        if gt != args.frames or gd != args.features:
            print(f"[Aviso] Input del modelo parece ser [{gt}, {gd}]. Usando parámetros CLI [{args.frames}, {args.features}].", file=sys.stderr)

    # Si pasás words.json, verifico dimensión de salida
    if args.words:
        labels = load_word_ids(args.words)
        if labels is not None and hasattr(model, "output_shape"):
            out_dim = model.output_shape[-1]
            if out_dim != len(labels):
                print(f"[Aviso] Clases del modelo ({out_dim}) != len(words.json) ({len(labels)})", file=sys.stderr)

    # Convierte
    tflite_bytes = convert_to_tflite(
        model=model,
        out_path=args.out,
        quant=args.quant,
        frames=args.frames,
        features=args.features,
        rep_path=args.rep,
        use_select_tf_ops=args.select_tf_ops,  # plan B opcional
    )

    # Verificación rápida (opcional)
    if args.verify:
        verify_equivalence(
            model, tflite_bytes,
            frames=args.frames, features=args.features, n=16
        )

if __name__ == "__main__":
    main()
