"""
Prepara el entrenamiento de la cabeza en el teléfono (PY-F4 export + PY-F8 fixture).

1. Entrena el modelo con todos los datos MENOS la tanda 2 de P1 (y su bloque "nada"):
   así la demo en la tablet de P1 evalúa con tomas que el modelo nunca vio.
2. Lo parte en dos:
   - backbone.tflite: las dos LSTM (sin Flex), de (1, 15, 126) al embedding de 128;
   - cabeza_inicial.json: los pesos de Dense64 → Dense64 → Dense C, con su L2 y el
     dropout de entrada, que la app entrena en Kotlin puro (CabezaDensa.kt).
3. Fija la paridad:
   - paridad_cabeza.json: embeddings + etiquetas + máscara de dropout → probabilidades,
     pérdida, gradientes y pesos después de 1 y 3 pasos de Adam, calculados con
     TensorFlow. Kotlin tiene que dar lo mismo (< 1e-5).
   - paridad_backbone.json: secuencias → embedding, con el TFLite de Python.
4. Corre en Python la misma personalización que va a hacer la tablet (ajuste con la
   tanda 1 de P1, evaluación en su tanda 2), como referencia para comparar.

Escribe en runs/<fecha>_cabeza/. Después se copian a la app:
  backbone.tflite + cabeza_inicial.json -> app/src/main/assets/federado/
  paridad_cabeza.json -> app/src/test/resources/
  paridad_backbone.json -> app/src/androidTest/assets/

Uso, desde la raíz de modelo_LSA:
    python federated/exportar_cabeza.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import date

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, RAIZ)
sys.path.insert(1, os.path.join(RAIZ, "scripts"))

PERSONA_DEMO = "P1"
L2 = 0.001
DROPOUT = 0.5
LR = 1e-3


def separar(modelo):
    """Backbone (LSTM, Dropout, LSTM) y las tres Dense de la cabeza."""
    import tensorflow as tf
    capas = modelo.layers
    assert [type(c).__name__ for c in capas] == ["LSTM", "Dropout", "LSTM", "Dropout", "Dense", "Dense", "Dense"]
    config = modelo.get_config()
    config["layers"] = config["layers"][:4] if config["layers"][0]["class_name"] != "InputLayer" \
        else config["layers"][:5]
    backbone = tf.keras.Sequential.from_config(config)
    backbone.build(modelo.input_shape)
    for origen, destino in zip(capas[:4], backbone.layers, strict=True):
        destino.set_weights(origen.get_weights())
    return backbone, capas[4:]


def cabeza_keras(densas, n_clases):
    """La cabeza sola, con los mismos pesos y la misma L2, para calcular la verdad."""
    import tensorflow as tf
    entrada = tf.keras.Input((128,))
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2))(entrada)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(L2))(x)
    salida = tf.keras.layers.Dense(n_clases, activation="softmax")(x)
    m = tf.keras.Model(entrada, salida)
    for capa, densa in zip(m.layers[1:], densas, strict=True):
        capa.set_weights(densa.get_weights())
    return m


def pasos_adam(m, emb, y1h, mascara, pasos):
    """Pérdida, gradientes y pesos después de `pasos` pasos de Adam (TF, legacy)."""
    import tensorflow as tf
    opt = tf.keras.optimizers.Adam(learning_rate=LR)  # en TF 2.10 es el Adam "legacy"
    x = tf.constant(emb * mascara)  # la máscara ya trae el reescalado 1/(1-p)
    historia = []
    for _ in range(pasos):
        with tf.GradientTape() as tape:
            p = m(x, training=True)
            ce = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(y1h, p))
            perdida = ce + tf.add_n(m.losses)
        grads = tape.gradient(perdida, m.trainable_variables)
        historia.append({"perdida": float(perdida.numpy()), "gradientes": [g.numpy().tolist() for g in grads]})
        opt.apply_gradients(zip(grads, m.trainable_variables, strict=True))
    return historia, [v.numpy().tolist() for v in m.trainable_variables]


def a_json_cabeza(densas, clases):
    capas = []
    for i, d in enumerate(densas):
        w, b = d.get_weights()
        capas.append({"W": w.tolist(), "b": b.tolist(), "activacion": "softmax" if i == 2 else "relu",
                      "l2": 0.0 if i == 2 else L2})
    return {"clases": clases, "dropout_entrada": DROPOUT, "capas": capas}


def main():
    import tensorflow as tf
    from validar import entrenar

    from lsa import dataset as ds
    from lsa.export import convertir, inferir_tflite, ops_flex

    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.utils.set_random_seed(0)

    salida = os.path.join(RAIZ, "runs", f"{date.today().isoformat()}_cabeza")
    os.makedirs(salida, exist_ok=True)
    d = ds.cargar(incluir_nada=True)
    C = len(d.clases)
    reservada = (d.persona == PERSONA_DEMO) & (d.sesion == f"{PERSONA_DEMO}-A2")
    print(d.resumen(), "| reservadas para la demo:", int(reservada.sum()), flush=True)
    modelo = entrenar(d.X[~reservada], d.y[~reservada], C, 0, 300)

    backbone, densas = separar(modelo)
    tfl = convertir(backbone, 15, 126, "f32")
    assert not ops_flex(tfl)
    with open(os.path.join(salida, "backbone.tflite"), "wb") as f:
        f.write(tfl)
    with open(os.path.join(salida, "cabeza_inicial.json"), "w", encoding="utf-8") as f:
        json.dump(a_json_cabeza(densas, d.clases), f)

    # Paridad del backbone: TFLite (lo que corre en el teléfono) contra Keras.
    tel = np.where(d.fuente == "telefono")[0]
    muestra = tel[:: max(1, len(tel) // 12)][:12]
    emb_keras = backbone.predict(d.X[muestra], verbose=0)
    emb_tfl = inferir_tflite(tfl, d.X[muestra])
    dif_backbone = float(np.abs(emb_keras - emb_tfl).max())
    with open(os.path.join(salida, "paridad_backbone.json"), "w", encoding="utf-8") as f:
        json.dump({"backbone_sha256": hashlib.sha256(tfl).hexdigest(), "tolerancia": 1e-4,
                   "casos": [{"entrada": np.round(d.X[i], 7).tolist(), "embedding": e.tolist()}
                             for i, e in zip(muestra, inferir_tflite(tfl, np.round(d.X[muestra], 7)
                                                                         .astype(np.float32)), strict=True)]}, f)

    # Paridad de la cabeza: un lote de 8 embeddings reales con una máscara de dropout fija.
    rng = np.random.default_rng(0)
    lote = emb_tfl[:8].astype(np.float32)
    y_lote = d.y[muestra][:8]
    y1h = tf.keras.utils.to_categorical(y_lote, C).astype(np.float32)
    mascara = (rng.random(lote.shape) >= DROPOUT).astype(np.float32) / (1 - DROPOUT)
    cab = cabeza_keras(densas, C)
    probs = cab(lote, training=False).numpy()
    historia, pesos_3 = pasos_adam(cab, lote, y1h, mascara, 3)
    cab1 = cabeza_keras(densas, C)
    _, pesos_1 = pasos_adam(cab1, lote, y1h, mascara, 1)
    with open(os.path.join(salida, "paridad_cabeza.json"), "w", encoding="utf-8") as f:
        json.dump({"lr": LR, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-7, "tolerancia": 1e-5,
                   "cabeza": a_json_cabeza(densas, d.clases), "embeddings": lote.tolist(),
                   "etiquetas": y_lote.tolist(), "mascara_dropout": mascara.tolist(),
                   "probs_sin_dropout": probs.tolist(), "paso1": historia[0], "pesos_tras_1": pesos_1,
                   "pesos_tras_3": pesos_3, "perdidas": [h["perdida"] for h in historia]}, f)

    # Referencia en Python de la personalización que va a correr la tablet.
    t1 = (d.persona == PERSONA_DEMO) & (d.sesion == f"{PERSONA_DEMO}-A1") & (d.fuente == "telefono")
    t2 = reservada & (d.fuente == "telefono")
    e1, e2 = backbone.predict(d.X[t1], verbose=0), backbone.predict(d.X[t2], verbose=0)
    ref = cabeza_keras(densas, C)
    ref.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="categorical_crossentropy", metrics=["accuracy"])
    antes = float((ref.predict(e2, verbose=0).argmax(1) == d.y[t2]).mean())
    drop = tf.keras.Sequential([tf.keras.layers.Dropout(DROPOUT), ref])
    drop.compile(optimizer=tf.keras.optimizers.Adam(LR), loss="categorical_crossentropy")
    drop.fit(e1, tf.keras.utils.to_categorical(d.y[t1], C), epochs=10, batch_size=8, verbose=0)
    despues = float((ref.predict(e2, verbose=0).argmax(1) == d.y[t2]).mean())

    info = {"datos": d.resumen(), "persona_demo": PERSONA_DEMO, "reservadas": int(reservada.sum()),
            "tanda1": int(t1.sum()), "tanda2": int(t2.sum()), "backbone_kb": round(len(tfl) / 1024, 1),
            "backbone_dif_max_keras_tflite": dif_backbone,
            "parametros_cabeza": int(sum(np.prod(v.shape) for v in cab.trainable_variables)),
            "python_personalizacion": {"antes": antes, "despues": despues, "epocas": 10, "batch": 8, "lr": LR}}
    with open(os.path.join(salida, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
