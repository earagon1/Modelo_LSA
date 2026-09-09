"""
Experimento comparativo: centralizado vs. aislado vs. FedAvg.

Genera la tabla del capitulo de resultados. Las cuatro filas contestan cuatro
preguntas distintas y solo juntas dicen algo:

  Centralizado   Techo teorico. Todos los datos en un solo lugar. Es lo que
                 FedAvg intenta igualar sin mover los datos.

  Solo local     Piso. Un cliente entrena unicamente con lo suyo, sin
                 colaborar. Es la comparacion que importa: si FedAvg no le
                 gana a esto, el enfoque federado no aporta nada.

  FedAvg IID     Reparto uniforme. El caso amable.

  FedAvg non-IID Reparto Dirichlet: cada cliente tiene un subconjunto sesgado
                 del vocabulario. Es el caso realista, y que rinda peor que el
                 IID es un resultado esperado, no una falla.

Todos los escenarios reciben el MISMO presupuesto de epocas (rounds x
local_epochs) y se evaluan sobre el MISMO conjunto de test, que ningun cliente
ve nunca. Sin esas dos condiciones la tabla no compara nada.

REPETICIONES
------------
El conjunto de test tiene ~184 muestras, asi que una sola muestra vale 0.54 %
de accuracy y la curva por ronda oscila varios puntos. Con una unica corrida,
diferencias de 2 o 3 puntos entre escenarios son indistinguibles del ruido.

Por eso el experimento se repite con varias semillas (`--repeticiones`) y se
reporta media +/- desvio. Cada replica resortea el split train/test, las
particiones y la inicializacion del modelo, o sea que el desvio estima la
variabilidad total del procedimiento. Sin esto la tabla no es defendible.

Uso:
    python federated/run_experiment.py --repeticiones 5     # recomendado
    python federated/run_experiment.py --clients 5 --alpha 0.1 --repeticiones 5
    python federated/run_experiment.py --quick              # prueba de humo
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

from data import (  # noqa: E402
    cargar_dataset,
    partition_dirichlet,
    partition_iid,
    reporte_particion,
    split_train_test,
)
from fedavg import run_fedavg, train_centralizado  # noqa: E402
from metrics import (  # noqa: E402
    classification_metrics,
    consistency_sigma,
    matriz_confusion,
    sugerir_epsilon,
)
from model import build_model, contar_parametros  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "resultados"

CENTRALIZADO = "Centralizado (techo)"
SOLO_LOCAL = "Solo local (piso)"


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--rounds", type=int, default=30, help="Rondas de FedAvg (default 30)")
    p.add_argument("--local-epochs", type=int, default=2, help="Epocas locales por ronda (default 2)")
    p.add_argument("--clients", type=int, default=3, help="Cantidad de clientes (default 3)")
    p.add_argument("--alpha", type=float, default=0.5, help="Alpha de Dirichlet, menor = mas non-IID (default 0.5)")
    p.add_argument("--repeticiones", type=int, default=1, help="Replicas con distinta semilla (default 1, recomendado 5)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--client-fraction", type=float, default=1.0, help="Fraccion C de clientes por ronda")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--eps-percentil", type=float, default=5.0, help="Percentil para el eps de la metrica sigma")
    p.add_argument("--seed", type=int, default=42, help="Semilla base; la replica i usa seed+i")
    p.add_argument("--quick", action="store_true", help="Corrida corta para verificar que todo enlaza")
    p.add_argument("--salida", type=Path, default=SALIDA)
    p.add_argument(
        "--reanalizar", type=Path, default=None,
        help="Recalcula tabla y lectura desde el resultados.json de una corrida previa, sin reentrenar",
    )
    return p.parse_args()


def reanalizar(directorio: Path):
    """
    Rehace la tabla y la lectura desde una corrida ya guardada.

    Sirve para cambiar el analisis sin volver a entrenar media hora, que es lo
    que hace falta cada vez que se corrige el criterio de interpretacion.
    """
    datos = json.loads((directorio / "resultados.json").read_text(encoding="utf-8"))
    replicas = datos["replicas"]
    agregado = agregar(replicas)

    tabla = construir_tabla(agregado, len(replicas))
    lectura = interpretar(agregado, replicas)

    print(tabla)
    print(lectura)

    (directorio / "tabla_resultados.md").write_text(
        tabla + "\n\n```\n" + lectura + "\n```\n", encoding="utf-8"
    )
    datos["agregado"] = agregado
    (directorio / "resultados.json").write_text(
        json.dumps(datos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nActualizado {directorio / 'tabla_resultados.md'}")


def evaluar(modelo, X_test, y_test_idx, eps, etiquetas, nombre_png=None, salida=None):
    """Metricas completas de un modelo ya entrenado."""
    probs = modelo.predict(X_test, batch_size=64, verbose=0)

    m = classification_metrics(y_test_idx, probs)
    cons = consistency_sigma(X_test, probs, eps)
    m["sigma"] = cons["sigma"]
    m["n_con_vecindad"] = cons["n_con_vecindad"]

    if nombre_png and salida:
        matriz_confusion(
            y_test_idx, probs, etiquetas, destino=salida / f"confusion_{nombre_png}.png"
        )
    return m


def correr_replica(args, seed: int, guardar_png: bool, verbose: bool):
    """
    Una replica completa de los cuatro escenarios con una semilla dada.

    Devuelve (resultados, curvas, info) donde `resultados` mapea nombre de
    escenario a sus metricas.
    """
    tf.keras.utils.set_random_seed(seed)
    presupuesto = args.rounds * args.local_epochs

    X, y, etiquetas = cargar_dataset(verbose=verbose)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=args.test_size, seed=seed
    )
    n_classes = len(etiquetas)
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes).astype(np.float32)
    y_test_oh = tf.keras.utils.to_categorical(y_test, n_classes).astype(np.float32)

    def construir():
        return build_model(X.shape[1], X.shape[2], n_classes)

    # eps unico para los cuatro escenarios de esta replica: si cada modelo
    # usara el suyo, los sigma no serian comparables entre si.
    eps = sugerir_epsilon(X_test, percentil=args.eps_percentil, seed=seed)

    part_iid = partition_iid(y_train, args.clients, seed=seed)
    part_niid = partition_dirichlet(y_train, args.clients, alpha=args.alpha, seed=seed)

    if verbose:
        print(f"\ntrain={len(X_train)}  test={len(X_test)}  clases={n_classes}")
        print(f"eps para sigma (percentil {args.eps_percentil}): {eps:.4f}")
        print("\n--- Particion IID ---")
        print(reporte_particion(part_iid, y_train, etiquetas))
        print(f"\n--- Particion non-IID (Dirichlet alpha={args.alpha}) ---")
        print(reporte_particion(part_niid, y_train, etiquetas))

    resultados, curvas = {}, {}

    # ------------------------------------------------------------- 1. techo
    t0 = time.time()
    cen = train_centralizado(
        construir, X_train, y_train_oh, X_test, y_test_oh,
        epochs=presupuesto, batch_size=args.batch_size, verbose=False,
    )
    resultados[CENTRALIZADO] = evaluar(
        cen["modelo"], X_test, y_test, eps, etiquetas,
        "centralizado" if guardar_png else None, args.salida,
    )
    resultados[CENTRALIZADO].update({"rondas": None, "mb": 0.0, "segundos": time.time() - t0})
    print(f"  centralizado      acc={resultados[CENTRALIZADO]['accuracy']:.4f}")

    # -------------------------------------------------------------- 2. piso
    idx0 = part_iid[0]
    t0 = time.time()
    solo = train_centralizado(
        construir, X_train[idx0], y_train_oh[idx0], X_test, y_test_oh,
        epochs=presupuesto, batch_size=args.batch_size, verbose=False,
    )
    resultados[SOLO_LOCAL] = evaluar(
        solo["modelo"], X_test, y_test, eps, etiquetas,
        "solo_local" if guardar_png else None, args.salida,
    )
    resultados[SOLO_LOCAL].update({"rondas": None, "mb": 0.0, "segundos": time.time() - t0})
    print(f"  solo local        acc={resultados[SOLO_LOCAL]['accuracy']:.4f}  "
          f"({len(idx0)} de {len(X_train)} muestras)")

    # ---------------------------------------------------------- 3. y 4. fed
    for nombre, particiones, slug in [
        (f"FedAvg IID ({args.clients} clientes)", part_iid, "fedavg_iid"),
        (f"FedAvg non-IID (alpha={args.alpha})", part_niid, "fedavg_niid"),
    ]:
        t0 = time.time()
        fed = run_fedavg(
            construir, X_train, y_train_oh, particiones, X_test, y_test_oh,
            rounds=args.rounds, local_epochs=args.local_epochs,
            batch_size=args.batch_size, client_fraction=args.client_fraction,
            seed=seed, verbose=False,
        )
        resultados[nombre] = evaluar(
            fed["modelo"], X_test, y_test, eps, etiquetas,
            slug if guardar_png else None, args.salida,
        )
        resultados[nombre].update({
            "rondas": args.rounds,
            "mb": fed["bytes_totales"] / 1e6,
            "segundos": time.time() - t0,
        })
        curvas[nombre] = fed["historia"]
        print(f"  {slug:16s}  acc={resultados[nombre]['accuracy']:.4f}")

    info = {
        "eps": eps,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "etiquetas": etiquetas,
        "n_params": contar_parametros(construir()),
    }
    return resultados, curvas, info


def agregar(replicas: list[dict]) -> dict:
    """Media y desvio de cada metrica sobre las replicas."""
    escenarios = list(replicas[0].keys())
    metricas = ["accuracy", "f1_macro", "f1_micro", "sigma"]

    agregado = {}
    for esc in escenarios:
        fila = {}
        for met in metricas:
            valores = np.array([r[esc][met] for r in replicas], dtype=float)
            fila[met] = float(np.nanmean(valores))
            fila[f"{met}_std"] = float(np.nanstd(valores, ddof=0))
        fila["rondas"] = replicas[0][esc]["rondas"]
        fila["mb"] = replicas[0][esc]["mb"]
        fila["n"] = len(replicas)
        agregado[esc] = fila
    return agregado


def construir_tabla(agregado: dict, n_reps: int) -> str:
    if n_reps > 1:
        cab = (
            f"Media +/- desvio sobre {n_reps} replicas con distinta semilla.\n\n"
            "| Escenario | Accuracy | F1-macro | F1-micro | sigma | Rondas | MB transferidos |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        fmt = lambda m, k: f"{m[k]:.4f} +/- {m[k + '_std']:.4f}"  # noqa: E731
    else:
        cab = (
            "Corrida unica: los valores NO tienen estimacion de variabilidad y no\n"
            "deben compararse entre si. Usar --repeticiones 5.\n\n"
            "| Escenario | Accuracy | F1-macro | F1-micro | sigma | Rondas | MB transferidos |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        fmt = lambda m, k: f"{m[k]:.4f}"  # noqa: E731

    filas = []
    for nombre, m in agregado.items():
        rondas = "-" if m["rondas"] is None else str(m["rondas"])
        mb = "-" if m["mb"] == 0 else f"{m['mb']:.1f}"
        filas.append(
            f"| {nombre} | {fmt(m, 'accuracy')} | {fmt(m, 'f1_macro')} | "
            f"{fmt(m, 'f1_micro')} | {fmt(m, 'sigma')} | {rondas} | {mb} |"
        )
    return cab + "\n".join(filas)


def _comparar(replicas: list[dict], a: str, b: str, alpha: float = 0.05) -> dict:
    """
    Compara dos escenarios con un t-test pareado sobre las replicas.

    El test es PAREADO porque dentro de una replica los cuatro escenarios
    comparten semilla: mismo split train/test, misma particion, misma
    inicializacion. Esa estructura hay que aprovecharla, porque el ruido
    comun a los cuatro se cancela y el test gana muchisima potencia.

    Comparar las medias contra la suma de los desvios crudos, que es lo que
    hacia la primera version de esta funcion, es un criterio demasiado
    conservador: confunde la dispersion entre replicas con la incertidumbre
    de la diferencia, que es sqrt(n) veces menor.
    """
    from scipy import stats

    va = np.array([r[a]["accuracy"] for r in replicas], dtype=float)
    vb = np.array([r[b]["accuracy"] for r in replicas], dtype=float)
    dif = va - vb

    n = len(dif)
    media = float(dif.mean())
    ee = float(dif.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")

    if n < 2 or np.allclose(dif, dif[0]):
        p = float("nan")
    else:
        p = float(stats.ttest_rel(va, vb).pvalue)

    return {
        "dif": media,
        "ee": ee,
        "p": p,
        "n": n,
        "significativo": bool(p == p and p < alpha),  # p == p descarta NaN
    }


def interpretar(agregado: dict, replicas: list[dict]) -> str:
    """
    Lectura automatica de la tabla: dice si las diferencias se sostienen.

    No reemplaza el analisis del informe, pero evita leer como conclusion algo
    que esta dentro del margen de error.
    """
    lineas = ["", "LECTURA DE LOS RESULTADOS", "=" * 72]
    n_reps = len(replicas)

    if n_reps < 2:
        lineas.append(
            "Con una sola replica no hay estimacion de variabilidad.\n"
            "Volve a correr con --repeticiones 5 antes de sacar conclusiones."
        )
        return "\n".join(lineas)

    lineas.append(f"t-test pareado sobre {n_reps} replicas, alpha=0.05.")
    lineas.append("")

    fed = [k for k in agregado if k.startswith("FedAvg")]

    lineas.append("Contra el PISO (cliente aislado) -- es la comparacion que")
    lineas.append("justifica el enfoque federado:")
    for nombre in fed:
        c = _comparar(replicas, nombre, SOLO_LOCAL)
        if c["significativo"] and c["dif"] > 0:
            v = "SUPERA al cliente aislado"
        elif c["significativo"]:
            v = "es PEOR que el cliente aislado"
        else:
            v = "no se distingue del cliente aislado"
        lineas.append(
            f"  {nombre}: {v} ({c['dif']:+.4f} +/- {c['ee']:.4f}, p={c['p']:.4f})"
        )

    lineas.append("")
    lineas.append("Contra el TECHO (centralizado) -- cuanto cuesta no centralizar:")
    for nombre in fed:
        c = _comparar(replicas, nombre, CENTRALIZADO)
        if not c["significativo"]:
            v = "alcanza al centralizado (diferencia no significativa)"
        elif c["dif"] < 0:
            v = f"queda {abs(c['dif']) * 100:.1f} puntos por debajo del centralizado"
        else:
            v = (
                "supera al centralizado. Es posible (el promediado periodico "
                "regulariza), pero conviene verificar que el centralizado no "
                "este sobreajustando"
            )
        lineas.append(
            f"  {nombre}: {v} ({c['dif']:+.4f} +/- {c['ee']:.4f}, p={c['p']:.4f})"
        )

    if len(fed) == 2:
        lineas.append("")
        lineas.append("IID contra non-IID -- cuanto cuesta la heterogeneidad:")
        c = _comparar(replicas, fed[0], fed[1])
        if c["significativo"]:
            v = f"el reparto non-IID cuesta {abs(c['dif']) * 100:.1f} puntos"
        else:
            v = "la diferencia entre IID y non-IID no es significativa"
        lineas.append(f"  {v} ({c['dif']:+.4f} +/- {c['ee']:.4f}, p={c['p']:.4f})")

    if n_reps < 5:
        lineas.append("")
        lineas.append(
            f"AVISO: {n_reps} replicas dan poca potencia estadistica. Un 'no "
            "significativo' aca puede ser falta de datos y no ausencia de efecto."
        )

    return "\n".join(lineas)


def graficar_curvas(curvas_por_replica: list[dict], agregado: dict, destino: Path):
    if not curvas_por_replica or not curvas_por_replica[0]:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    nombres = list(curvas_por_replica[0].keys())

    for nombre in nombres:
        acc = np.array([r[nombre]["accuracy"] for r in curvas_por_replica])
        rondas = curvas_por_replica[0][nombre]["ronda"]
        media = acc.mean(axis=0)
        linea, = ax.plot(rondas, media, marker="o", markersize=3, label=nombre)
        if len(acc) > 1:
            desvio = acc.std(axis=0)
            ax.fill_between(
                rondas, media - desvio, media + desvio,
                alpha=0.18, color=linea.get_color(),
            )

    for nombre, estilo in [(CENTRALIZADO, "--"), (SOLO_LOCAL, ":")]:
        if nombre in agregado:
            ax.axhline(
                agregado[nombre]["accuracy"], linestyle=estilo, color="gray",
                linewidth=1.4,
                label=f"{nombre} = {agregado[nombre]['accuracy']:.3f}",
            )

    n = len(curvas_por_replica)
    ax.set_xlabel("Ronda de comunicacion")
    ax.set_ylabel("Accuracy sobre el test global")
    ax.set_title(
        f"FedAvg: accuracy por ronda contra techo y piso"
        + (f"  (media +/- desvio, {n} replicas)" if n > 1 else "")
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()

    if args.reanalizar is not None:
        reanalizar(args.reanalizar)
        return

    if args.quick:
        args.rounds, args.local_epochs, args.repeticiones = 3, 1, 1

    args.salida.mkdir(parents=True, exist_ok=True)
    presupuesto = args.rounds * args.local_epochs

    print("=" * 72)
    print("EXPERIMENTO FEDERADO -- LSA")
    print("=" * 72)
    print(f"clientes={args.clients}  rondas={args.rounds}  epocas locales={args.local_epochs}")
    print(f"presupuesto de epocas por escenario={presupuesto}  batch={args.batch_size}")
    print(f"alpha Dirichlet={args.alpha}  C={args.client_fraction}")
    print(f"replicas={args.repeticiones}  semilla base={args.seed}")
    print("=" * 72)

    replicas, curvas_por_replica, info = [], [], None
    t_total = time.time()

    for i in range(args.repeticiones):
        seed = args.seed + i
        print(f"\n--- Replica {i + 1}/{args.repeticiones} (seed={seed}) ---")
        res, cur, info = correr_replica(
            args, seed, guardar_png=(i == 0), verbose=(i == 0)
        )
        replicas.append(res)
        curvas_por_replica.append(cur)

    agregado = agregar(replicas)
    tabla = construir_tabla(agregado, args.repeticiones)
    lectura = interpretar(agregado, replicas)

    print("\n" + "=" * 72)
    print(tabla)
    print(lectura)
    print(f"\nTiempo total: {(time.time() - t_total) / 60:.1f} min")

    (args.salida / "tabla_resultados.md").write_text(
        tabla + "\n\n```\n" + lectura + "\n```\n", encoding="utf-8"
    )

    config = vars(args).copy()
    config["salida"] = str(config["salida"])
    config.update(info)
    (args.salida / "resultados.json").write_text(
        json.dumps(
            {
                "config": config,
                "agregado": agregado,
                "replicas": replicas,
                "curvas": curvas_por_replica,
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    graficar_curvas(curvas_por_replica, agregado, args.salida / "curva_accuracy.png")

    print(f"\nGuardado en {args.salida}")
    print("  tabla_resultados.md   tabla y lectura para el informe")
    print("  resultados.json       metricas por replica y curvas completas")
    print("  curva_accuracy.png    accuracy vs ronda, con banda de desvio")
    print("  confusion_*.png       matriz de confusion (primera replica)")


if __name__ == "__main__":
    main()
