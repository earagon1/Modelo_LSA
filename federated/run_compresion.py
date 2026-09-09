"""
Experimento de compresion: cuanto se puede achicar la subida sin perder modelo.

FedAvg sin comprimir movio 193 MB en la corrida de 5 clientes por 30 rondas.
Para un escenario movil eso es prohibitivo, y es el problema que Konecny et al.
(2017) atacan con actualizaciones esbozadas.

Este script barre cuantos bits se le dedican a cada valor del delta que el
cliente sube, y mide que pasa con la accuracy. La pregunta concreta es donde
esta el punto en que ahorrar mas empieza a costar modelo.

Ademas compara redondeo estocastico contra determinista al mismo numero de
bits. Esa comparacion aisla una sola cosa: si el sesgo del redondeo importa
cuando se promedian K clientes ronda tras ronda.

Todas las configuraciones comparten datos, particiones, semillas y presupuesto
de epocas; lo unico que cambia es como se codifica el update. Se corren varias
replicas porque, como quedo claro en `run_experiment.py`, con 184 muestras de
test una diferencia de dos o tres puntos es ruido.

Uso:
    python federated/run_compresion.py                      # 5 clientes, 5 replicas
    python federated/run_compresion.py --repeticiones 3 --rounds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import tensorflow as tf  # noqa: E402

from data import cargar_dataset, partition_iid, split_train_test  # noqa: E402
from fedavg import run_fedavg  # noqa: E402
from metrics import classification_metrics  # noqa: E402
from model import build_model  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "resultados_compresion"

# (etiqueta, bits, estocastico). bits=None es el camino sin comprimir.
CONFIGS = [
    ("sin comprimir (float32)", None, False),
    ("8 bits estocastico", 8, True),
    ("4 bits estocastico", 4, True),
    ("2 bits estocastico", 2, True),
    ("4 bits determinista", 4, False),
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=2)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--repeticiones", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", type=Path, default=SALIDA)
    return p.parse_args()


def main():
    args = parse_args()
    args.salida.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("COMPRESION DE ACTUALIZACIONES FEDERADAS")
    print("=" * 72)
    print(f"clientes={args.clients}  rondas={args.rounds}  epocas locales={args.local_epochs}")
    print(f"replicas={args.repeticiones}  semilla base={args.seed}")
    print("=" * 72)

    replicas = []
    curvas = []
    t_total = time.time()

    for rep in range(args.repeticiones):
        seed = args.seed + rep
        print(f"\n--- Replica {rep + 1}/{args.repeticiones} (seed={seed}) ---")
        tf.keras.utils.set_random_seed(seed)

        X, y, etiquetas = cargar_dataset(verbose=False)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.2, seed=seed)
        n_classes = len(etiquetas)
        y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes).astype(np.float32)
        y_test_oh = tf.keras.utils.to_categorical(y_test, n_classes).astype(np.float32)

        particiones = partition_iid(y_train, args.clients, seed=seed)

        def construir():
            return build_model(X.shape[1], X.shape[2], n_classes)

        resultados_rep = {}
        curvas_rep = {}

        for etiqueta, bits, estocastico in CONFIGS:
            t0 = time.time()
            fed = run_fedavg(
                construir, X_train, y_train_oh, particiones, X_test, y_test_oh,
                rounds=args.rounds, local_epochs=args.local_epochs,
                batch_size=args.batch_size, seed=seed,
                bits=bits, estocastico=estocastico, verbose=False,
            )

            probs = fed["modelo"].predict(X_test, batch_size=64, verbose=0)
            m = classification_metrics(y_test, probs)
            m["mb_subida"] = fed["bytes_subida"] / 1e6
            m["mb_bajada"] = fed["bytes_bajada"] / 1e6
            m["mb_total"] = fed["bytes_totales"] / 1e6
            m["error_compresion"] = float(np.mean(fed["historia"]["error_compresion"]))
            m["segundos"] = time.time() - t0

            resultados_rep[etiqueta] = m
            curvas_rep[etiqueta] = fed["historia"]

            print(f"  {etiqueta:26s} acc={m['accuracy']:.4f}  "
                  f"subida={m['mb_subida']:6.1f} MB  "
                  f"err={m['error_compresion']:.4f}")

            tf.keras.backend.clear_session()

        replicas.append(resultados_rep)
        curvas.append(curvas_rep)

    agregado = agregar(replicas)
    tabla = construir_tabla(agregado, args.repeticiones)
    lectura = interpretar(agregado, replicas)

    print("\n" + "=" * 72)
    print(tabla)
    print(lectura)
    print(f"\nTiempo total: {(time.time() - t_total) / 60:.1f} min")

    (args.salida / "tabla_compresion.md").write_text(
        tabla + "\n\n```\n" + lectura + "\n```\n", encoding="utf-8"
    )
    config = {k: str(v) for k, v in vars(args).items()}
    (args.salida / "resultados.json").write_text(
        json.dumps({"config": config, "agregado": agregado,
                    "replicas": replicas, "curvas": curvas},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    graficar(agregado, args.salida / "accuracy_vs_bytes.png")

    print(f"\nGuardado en {args.salida}")


def agregar(replicas: list[dict]) -> dict:
    agregado = {}
    for etiqueta in replicas[0]:
        fila = {}
        for met in ["accuracy", "f1_macro", "error_compresion"]:
            v = np.array([r[etiqueta][met] for r in replicas], dtype=float)
            fila[met] = float(v.mean())
            fila[f"{met}_std"] = float(v.std(ddof=0))
        fila["mb_subida"] = replicas[0][etiqueta]["mb_subida"]
        fila["mb_total"] = replicas[0][etiqueta]["mb_total"]
        agregado[etiqueta] = fila
    return agregado


def construir_tabla(agregado: dict, n_reps: int) -> str:
    base = agregado["sin comprimir (float32)"]
    cab = (
        f"Media +/- desvio sobre {n_reps} replicas. Solo se comprime la subida;\n"
        f"la bajada del modelo global va siempre en float32.\n\n"
        "| Codificacion del update | MB subida | Ahorro | MB total | Accuracy | F1-macro | Error del delta |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    filas = []
    for etiqueta, m in agregado.items():
        ahorro = (1 - m["mb_subida"] / base["mb_subida"]) * 100
        ahorro_txt = "-" if abs(ahorro) < 0.01 else f"{ahorro:.0f} %"
        err = "-" if m["error_compresion"] == 0 else f"{m['error_compresion']:.4f}"
        filas.append(
            f"| {etiqueta} | {m['mb_subida']:.1f} | {ahorro_txt} | {m['mb_total']:.1f} | "
            f"{m['accuracy']:.4f} +/- {m['accuracy_std']:.4f} | "
            f"{m['f1_macro']:.4f} +/- {m['f1_macro_std']:.4f} | {err} |"
        )
    return cab + "\n".join(filas)


def interpretar(agregado: dict, replicas: list[dict]) -> str:
    from scipy import stats

    base_lbl = "sin comprimir (float32)"
    lineas = ["", "LECTURA DE LOS RESULTADOS", "=" * 72]

    if len(replicas) < 2:
        lineas.append("Con una sola replica no hay margen de error. Correr con --repeticiones 5.")
        return "\n".join(lineas)

    lineas.append(f"t-test pareado contra '{base_lbl}', {len(replicas)} replicas, alpha=0.05.")
    lineas.append("")

    base_vals = np.array([r[base_lbl]["accuracy"] for r in replicas])

    for etiqueta in agregado:
        if etiqueta == base_lbl:
            continue
        vals = np.array([r[etiqueta]["accuracy"] for r in replicas])
        dif = float((vals - base_vals).mean())
        p = float(stats.ttest_rel(vals, base_vals).pvalue)
        ahorro = (1 - agregado[etiqueta]["mb_subida"] / agregado[base_lbl]["mb_subida"]) * 100
        if p < 0.05:
            v = f"PIERDE {abs(dif) * 100:.1f} puntos" if dif < 0 else f"gana {dif * 100:.1f} puntos"
        else:
            v = "sin perdida medible"
        lineas.append(f"  {etiqueta:26s} ahorra {ahorro:.0f} % de subida, {v} (p={p:.4f})")

    # El redondeo: misma cantidad de bits, distinto sesgo.
    if "4 bits estocastico" in agregado and "4 bits determinista" in agregado:
        a = np.array([r["4 bits estocastico"]["accuracy"] for r in replicas])
        b = np.array([r["4 bits determinista"]["accuracy"] for r in replicas])
        p = float(stats.ttest_rel(a, b).pvalue)
        dif = float((a - b).mean())
        lineas.append("")
        lineas.append("Estocastico contra determinista, a igual cantidad de bits:")
        if p < 0.05:
            lineas.append(
                f"  el estocastico rinde {dif * 100:+.1f} puntos (p={p:.4f}): el sesgo del "
                "redondeo determinista no se cancela al promediar."
            )
        else:
            lineas.append(
                f"  no hay diferencia significativa ({dif * 100:+.1f} puntos, p={p:.4f}). "
                "Con esta cantidad de clientes y rondas el sesgo no alcanza a acumularse."
            )

    return "\n".join(lineas)


def graficar(agregado: dict, destino: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    for etiqueta, m in agregado.items():
        ax.errorbar(
            m["mb_subida"], m["accuracy"], yerr=m["accuracy_std"],
            fmt="o", markersize=9, capsize=4, label=etiqueta,
        )
        ax.annotate(
            etiqueta, (m["mb_subida"], m["accuracy"]),
            textcoords="offset points", xytext=(8, 6), fontsize=8.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("MB subidos en total (escala log)")
    ax.set_ylabel("Accuracy sobre el test global")
    ax.set_title("Cuanto cuesta ahorrar comunicacion")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
