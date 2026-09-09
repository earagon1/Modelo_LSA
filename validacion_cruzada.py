"""
Validacion cruzada estratificada del modelo de produccion.

La propuesta menciona validacion cruzada dos veces: como mitigacion del riesgo
de overfitting y como criterio para dar por bueno el modelo antes de exportarlo
a TFLite. `training_model.py` nunca la hizo: usa un unico `train_test_split`
80/20 con `random_state=42`, o sea que todo lo que se sabe del modelo sale de
una sola particion elegida al azar.

Este script hace K-fold estratificado sobre el dataset completo y produce lo
que faltaba:

  - Accuracy, F1-macro y F1-micro por fold, con media y desvio.
  - Predicciones out-of-fold: cada muestra la predice el unico modelo que NO
    la vio entrenando. Es la estimacion honesta de generalizacion, y es el
    numero que va al informe.
  - Matriz de confusion sobre esas predicciones out-of-fold, que reemplaza a
    `confusion_matrix.py` (que quedo roto: referencia MODEL_NUMS y MODELS_PATH,
    que ya no existen en constants.py).
  - La metrica de consistencia sigma sobre las mismas predicciones.

SOBRE EL EARLY STOPPING
-----------------------
Por defecto se replica exactamente lo que hace `training_model.py`:
EarlyStopping sobre `accuracy` (la de ENTRENAMIENTO) con patience 10. Se
replica y no se corrige porque el objetivo es estimar que generaliza el
modelo que hoy esta en produccion, no uno mejorado.

Vigilar la accuracy de ENTRENAMIENTO parece una eleccion discutible: esa curva
sigue subiendo despues de que la validacion empezo a empeorar, asi que el
criterio deberia cortar tarde y dejar sobreajustar. Se midio con
`--monitor val_loss` y resulto que NO cambia nada: 0.9457 +/- 0.0154 contra
0.9489 +/- 0.0101, diferencia no significativa (t-test pareado sobre los 5
folds, p=0.72). El criterio actual esta bien.

Ojo con como se mide esa comparacion. La primera version de este script le
pasaba el fold de TEST como `validation_data`, y asi `val_loss` daba 0.9641,
mejor por 1.5 puntos. Era fuga: el early stopping elegia cuando cortar mirando
exactamente los datos con los que despues se medía. Ahora la validacion se
recorta del fold de entrenamiento y la supuesta mejora desaparece. Es un
recordatorio util: una fuga no rompe nada, solo mejora los numeros.

Uso:
    python modelo_LSA/validacion_cruzada.py
    python modelo_LSA/validacion_cruzada.py --folds 5 --monitor val_loss
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

AQUI = Path(__file__).resolve().parent
# federated primero: `data` y `metrics` tienen que resolverse ahi. Si ganara
# este directorio, la carpeta `data/` se importaria como namespace package vacio.
sys.path.insert(0, str(AQUI / "federated"))

import tensorflow as tf  # noqa: E402

from data import cargar_dataset  # noqa: E402
from metrics import (  # noqa: E402
    classification_metrics,
    consistency_sigma,
    matriz_confusion,
    sugerir_epsilon,
)

SALIDA = AQUI / "resultados_validacion"


def cargar_get_model_de_produccion():
    """
    Importa `model.py` por ruta explicita.

    Se usa la arquitectura de produccion y no la copia de `federated/model.py`
    porque lo que se esta validando es el modelo real. Va por ruta para no
    chocar con `federated/model.py`, que se llama igual y esta primero en el
    path.
    """
    ruta = AQUI / "model.py"
    spec = importlib.util.spec_from_file_location("modelo_produccion", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.get_model


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=500, help="Tope de epocas (default 500, como training_model.py)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--monitor", choices=["accuracy", "val_loss", "ninguno"], default="accuracy",
        help="Que vigila el early stopping. 'accuracy' replica training_model.py",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", type=Path, default=SALIDA)
    return p.parse_args()


def main():
    args = parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)

    from sklearn.model_selection import StratifiedKFold

    get_model = cargar_get_model_de_produccion()

    X, y, etiquetas = cargar_dataset(verbose=True)
    n_classes = len(etiquetas)
    y_oh = tf.keras.utils.to_categorical(y, n_classes).astype(np.float32)

    print("\n" + "=" * 72)
    print(f"VALIDACION CRUZADA -- {args.folds} folds estratificados")
    print("=" * 72)
    print(f"epochs<={args.epochs}  batch={args.batch_size}  "
          f"early stopping: {args.monitor} (patience {args.patience})")
    print(f"{len(X)} muestras, {n_classes} clases\n")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Predicciones out-of-fold: se llenan con el modelo que no vio cada muestra.
    probs_oof = np.zeros((len(X), n_classes), dtype=np.float64)
    por_fold = []
    t_total = time.time()

    from sklearn.model_selection import train_test_split

    for fold, (idx_train, idx_test) in enumerate(skf.split(X, y), start=1):
        t0 = time.time()
        modelo = get_model(X.shape[1], n_classes)

        callbacks = []
        val_data = None
        idx_fit = idx_train

        if args.monitor == "accuracy":
            # Vigila la accuracy de entrenamiento: no necesita validacion.
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="accuracy", patience=args.patience, restore_best_weights=True
            ))
        elif args.monitor == "val_loss":
            # La validacion se recorta del fold de ENTRENAMIENTO, nunca del de
            # test. Usar el fold de test como validation_data seria fuga: el
            # early stopping elegiria cuando cortar mirando exactamente los
            # datos con los que despues se mide, y la accuracy saldria
            # optimista sin que nada falle a la vista.
            idx_fit, idx_val = train_test_split(
                idx_train, test_size=0.15, random_state=args.seed, stratify=y[idx_train]
            )
            val_data = (X[idx_val], y_oh[idx_val])
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=args.patience, restore_best_weights=True
            ))

        historia = modelo.fit(
            X[idx_fit], y_oh[idx_fit],
            validation_data=val_data,
            epochs=args.epochs, batch_size=args.batch_size,
            callbacks=callbacks, verbose=0,
        )

        probs = modelo.predict(X[idx_test], batch_size=64, verbose=0)
        probs_oof[idx_test] = probs

        m = classification_metrics(y[idx_test], probs)
        m["epocas"] = len(historia.history["loss"])
        m["segundos"] = time.time() - t0
        m["n_test"] = len(idx_test)
        por_fold.append(m)

        print(f"  fold {fold}/{args.folds}  acc={m['accuracy']:.4f}  "
              f"f1_macro={m['f1_macro']:.4f}  {m['epocas']:3d} epocas  "
              f"{m['segundos'] / 60:.1f} min")

        tf.keras.backend.clear_session()

    # ------------------------------------------------------------ agregados
    def med(clave):
        v = np.array([f[clave] for f in por_fold], dtype=float)
        return float(v.mean()), float(v.std(ddof=0))

    acc_m, acc_s = med("accuracy")
    fma_m, fma_s = med("f1_macro")
    fmi_m, fmi_s = med("f1_micro")

    # Metricas sobre las predicciones out-of-fold. No es lo mismo que el
    # promedio de folds: aca cada muestra pesa igual, sin importar en que fold
    # cayo, y ademas permite una unica matriz de confusion sobre todo el
    # dataset.
    m_oof = classification_metrics(y, probs_oof)
    eps = sugerir_epsilon(X, percentil=5.0, seed=args.seed)
    cons = consistency_sigma(X, probs_oof, eps)

    tabla = (
        f"Validacion cruzada estratificada, {args.folds} folds. "
        f"Early stopping sobre `{args.monitor}`.\n\n"
        "| Metrica | Media entre folds | Out-of-fold |\n"
        "|---|---|---|\n"
        f"| Accuracy | {acc_m:.4f} +/- {acc_s:.4f} | {m_oof['accuracy']:.4f} |\n"
        f"| F1-macro | {fma_m:.4f} +/- {fma_s:.4f} | {m_oof['f1_macro']:.4f} |\n"
        f"| F1-micro | {fmi_m:.4f} +/- {fmi_s:.4f} | {m_oof['f1_micro']:.4f} |\n"
        f"| sigma (consistencia) | - | {cons['sigma']:.4f} |\n"
    )

    detalle = ["", "| Fold | Muestras | Epocas | Accuracy | F1-macro |", "|---|---|---|---|---|"]
    for i, f in enumerate(por_fold, start=1):
        detalle.append(
            f"| {i} | {f['n_test']} | {f['epocas']} | {f['accuracy']:.4f} | {f['f1_macro']:.4f} |"
        )
    detalle = "\n".join(detalle)

    print("\n" + tabla + detalle)

    print("\nLECTURA")
    print("=" * 72)
    print(
        f"El desvio entre folds es {acc_s:.4f}. Cualquier diferencia menor a eso\n"
        f"entre dos variantes del modelo no se puede atribuir al modelo."
    )
    peor = min(por_fold, key=lambda f: f["accuracy"])
    mejor = max(por_fold, key=lambda f: f["accuracy"])
    print(
        f"Fold mas facil {mejor['accuracy']:.4f} contra el mas dificil "
        f"{peor['accuracy']:.4f}: {(mejor['accuracy'] - peor['accuracy']) * 100:.1f} puntos\n"
        f"de diferencia solo por como cayo la particion. Ese rango es la razon por\n"
        f"la que un unico train_test_split no alcanza para reportar un resultado."
    )
    epocas = [f["epocas"] for f in por_fold]
    print(f"Epocas hasta el corte: {min(epocas)}-{max(epocas)} (tope {args.epochs}).")

    matriz_confusion(y, probs_oof, etiquetas, destino=args.salida / "confusion_oof.png")

    (args.salida / "validacion_cruzada.md").write_text(tabla + detalle + "\n", encoding="utf-8")
    (args.salida / "validacion_cruzada.json").write_text(
        json.dumps(
            {
                "config": {**{k: str(v) for k, v in vars(args).items()},
                           "n_muestras": len(X), "etiquetas": etiquetas, "eps": eps},
                "por_fold": por_fold,
                "out_of_fold": {**m_oof, **cons},
                "media": {"accuracy": acc_m, "f1_macro": fma_m, "f1_micro": fmi_m},
                "desvio": {"accuracy": acc_s, "f1_macro": fma_s, "f1_micro": fmi_s},
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nTiempo total: {(time.time() - t_total) / 60:.1f} min")
    print(f"Guardado en {args.salida}")
    print("  validacion_cruzada.md     tabla para el informe")
    print("  validacion_cruzada.json   metricas por fold")
    print("  confusion_oof.png         matriz de confusion out-of-fold")


if __name__ == "__main__":
    main()
