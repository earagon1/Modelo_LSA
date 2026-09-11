import json, argparse, datetime
from pathlib import Path
from collections import Counter
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

def load_lsa_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    T, D = int(data.get("t", 15)), int(data.get("d", 126))
    X_list, y_list = [], []
    for _, arr in data.get("by_date", {}).items():
        for item in arr:
            seq = np.asarray(item["seq"], dtype=np.float32).reshape(T, D)
            X_list.append(seq); y_list.append(item["label"])
    X = np.stack(X_list, axis=0); y = np.array(y_list, dtype=object)
    return X, y

def ensure_label_map(labels):
    uniq = sorted(set(labels))
    lab2id = {lab:i for i,lab in enumerate(uniq)}
    id2lab = {i:lab for lab,i in lab2id.items()}
    return lab2id, id2lab

def build_or_adapt_model(base_model_path, input_shape, n_classes, unfreeze_tail=2):
    if base_model_path and Path(base_model_path).exists():
        m = tf.keras.models.load_model(base_model_path, compile=False)
        last = m.layers[-1]
        if not isinstance(last, tf.keras.layers.Dense) or last.units != n_classes:
            x = m.layers[-2].output
            out = tf.keras.layers.Dense(n_classes, activation="softmax", name="logits")(x)
            m = tf.keras.Model(m.input, out)
        # descongelar últimas 'unfreeze_tail' capas
        for lyr in m.layers[-unfreeze_tail:]:
            lyr.trainable = True
    else:
        inp = tf.keras.Input(shape=input_shape)  # (T,D)
        x = tf.keras.layers.Masking(mask_value=0.0)(inp)
        x = tf.keras.layers.LSTM(128, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.LSTM(128)(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(128, activation="relu")(x)
        m = tf.keras.Model(inp, tf.keras.layers.Dense(n_classes, activation="softmax")(x))
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m

#def to_tflite(h5_path: Path, out_path: Path):
#    model = tf.keras.models.load_model(h5_path, compile=False)
#    tfl = tf.lite.TFLiteConverter.from_keras_model(model).convert()
#    out_path.write_bytes(tfl)

def to_tflite(h5_path, out_path):
    import tensorflow as tf
    model = tf.keras.models.load_model(h5_path, compile=False)

    # Intento 1: builtin + desactivar lowering de TensorList
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        # clave para evitar el error que viste
        converter._experimental_lower_tensor_list_ops = False
        converter.experimental_enable_resource_variables = True
        converter.optimizations = []  # podés agregar DEFAULT si querés
        tfl = converter.convert()
        out_path.write_bytes(tfl)
        print("[OK] TFLite (builtin ops) generado:", out_path)
        return
    except Exception as e:
        print("[WARN] Builtin conversion falló, probando con Select TF Ops…", e)

    # Intento 2: habilitar Select TF Ops (Flex delegate)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter._experimental_lower_tensor_list_ops = False
    converter.experimental_enable_resource_variables = True
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,   # ← habilita TF ops
    ]
    tfl = converter.convert()
    out_path.write_bytes(tfl)
    print("[OK] TFLite (Select TF Ops) generado:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", default=["data/lsa_samples.json"], help="uno o varios lsa_samples.json")
    ap.add_argument("--base-model", type=str, default="models/actions_15.keras", help=".keras/.h5 opcional (NO /usando/)")
    ap.add_argument("--outdir", type=str, default=None, help="carpeta de salida")
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # salida por fecha/hora si no se especifica
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    outdir = Path(args.outdir) if args.outdir else Path("models/out") / f"finetune_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    # cargar JSON(s)
    Xs, ys = [], []
    for jp in args.json:
        Xi, yi = load_lsa_json(Path(jp)); Xs.append(Xi); ys.append(yi)
    X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
    print(f"[INFO] X={X.shape} | clases={Counter(y)}")

    T, D = X.shape[1], X.shape[2]
    lab2id, id2lab = ensure_label_map(y)
    (outdir / "label_map.json").write_text(json.dumps({"lab2id": lab2id, "id2lab": id2lab}, indent=2, ensure_ascii=False), "utf-8")
    y_ids = np.array([lab2id[v] for v in y], dtype=np.int64)

    Xtr, Xva, Ytr, Yva = train_test_split(X, y_ids, test_size=args.val_split, stratify=y_ids, random_state=42)
    classes = np.unique(Ytr)
    cw = compute_class_weight("balanced", classes=classes, y=Ytr)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw)}

    model = build_or_adapt_model(args.base_model, (T, D), len(lab2id))
    model.summary()

    cbs = [
        tf.keras.callbacks.ModelCheckpoint(str(outdir/"best.h5"), monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
              epochs=args.epochs, batch_size=args.batch,
              class_weight=class_weight, verbose=2, callbacks=cbs)

    h5_out = outdir / "model_finetuned.h5"
    model.save(h5_out)
    to_tflite(h5_out, outdir / "model_finetuned.tflite")

    val_loss, val_acc = model.evaluate(Xva, Yva, verbose=0)
    print(f"[RESULT] val_acc={val_acc:.4f}  val_loss={val_loss:.4f}")
    print(f"[OK] Guardado en: {outdir}")

if __name__ == "__main__":
    main()


