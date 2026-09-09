"""
Federated Averaging (FedAvg), McMahan et al. 2017.

El algoritmo entero es esto:

    para cada ronda r = 1..R:
        seleccionar K clientes de los N disponibles
        para cada cliente k:
            arrancar de los pesos globales
            entrenar E epocas sobre los datos locales de k
            devolver (pesos_k, n_k)          <- nunca datos crudos
        pesos_globales = SUM (n_k / n) * pesos_k

Lo unico que viaja son los pesos: los datos de cada cliente no salen nunca de
su particion. Esa es toda la propiedad que el enfoque promete.

`aggregate` esta escrita como funcion pura sobre listas de arrays justamente
para poder testearla sin entrenar nada (ver `test_fedavg.py`).
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from compresion import comprimir_update, error_relativo


def aggregate(updates: list[tuple[list[np.ndarray], int]]) -> list[np.ndarray]:
    """
    Promedio ponderado por cantidad de muestras: w = SUM (n_k / n) * w_k.

    La ponderacion importa: un cliente con 90 muestras debe pesar 9 veces mas
    que uno con 10. Si se promediara sin ponderar, un cliente con datos
    escasos tendria la misma influencia que uno con muchos y el modelo global
    se degradaria.

    Se acumula en float64 y se devuelve float32 para que la suma de fracciones
    no arrastre error cuando hay muchos clientes.
    """
    if not updates:
        raise ValueError("aggregate() recibio una lista vacia de updates")

    total = sum(n for _, n in updates)
    if total <= 0:
        raise ValueError("La suma de muestras de los clientes es cero")

    acumulado = [np.zeros_like(w, dtype=np.float64) for w in updates[0][0]]
    for pesos, n in updates:
        if len(pesos) != len(acumulado):
            raise ValueError("Los clientes devolvieron listas de pesos de distinto largo")
        fraccion = n / total
        for i, w in enumerate(pesos):
            acumulado[i] += fraccion * np.asarray(w, dtype=np.float64)

    return [a.astype(np.float32) for a in acumulado]


def weights_nbytes(pesos: list[np.ndarray]) -> int:
    """Tamano en bytes de un juego de pesos, asumiendo float32 en la red."""
    return int(sum(np.asarray(w).size for w in pesos) * 4)


def reset_optimizer_state(modelo: tf.keras.Model) -> None:
    """
    Pone en cero los momentos de Adam y el contador de pasos.

    Hace falta porque se reusa una sola instancia de modelo para todos los
    clientes: sin este reset, los momentos que Adam acumulo entrenando al
    cliente 0 condicionan el entrenamiento del cliente 1. Eso es una fuga de
    informacion entre clientes y rompe la premisa del esquema federado, ademas
    de hacer que el resultado dependa del orden en que se recorren.

    En FedAvg cada cliente es stateless entre rondas: recibe pesos, entrena
    desde cero en lo que a estado del optimizador respecta, y devuelve pesos.

    Este comportamiento lo verifica `test_identidad_un_cliente_equivale_a_centralizado`.
    """
    opt = getattr(modelo, "optimizer", None)
    if opt is None:
        return

    # API legacy (OptimizerV2, TF <= 2.10): incluye 'iterations' en get_weights().
    try:
        pesos = opt.get_weights()
        if pesos:
            opt.set_weights([np.zeros_like(w) for w in pesos])
        return
    except (AttributeError, NotImplementedError, ValueError):
        pass

    # API nueva (TF >= 2.11).
    variables = getattr(opt, "variables", None)
    if callable(variables):
        variables = variables()
    for v in variables or []:
        v.assign(tf.zeros_like(v))


def local_train(
    modelo: tf.keras.Model,
    pesos_globales: list[np.ndarray],
    X,
    y_onehot,
    epochs: int,
    batch_size: int = 8,
    shuffle: bool = True,
    verbose: int = 0,
) -> list[np.ndarray]:
    """
    Un round local: arrancar de los pesos globales y entrenar E epocas.

    Se reusa siempre la misma instancia de modelo (solo se le cambian los pesos)
    porque construir un Keras nuevo por cliente y por ronda es carisimo y llena
    el grafo de TF de basura. El precio de reusarla es tener que limpiar el
    estado del optimizador a mano, que es lo que hace `reset_optimizer_state`.
    """
    modelo.set_weights(pesos_globales)
    reset_optimizer_state(modelo)
    modelo.fit(
        X,
        y_onehot,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=shuffle,
        verbose=verbose,
    )
    return modelo.get_weights()


def run_fedavg(
    construir_modelo,
    X_train,
    y_train_onehot,
    particiones: list[np.ndarray],
    X_test,
    y_test_onehot,
    rounds: int = 30,
    local_epochs: int = 2,
    batch_size: int = 8,
    client_fraction: float = 1.0,
    seed: int = 42,
    bits: int | None = None,
    estocastico: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Corre FedAvg y devuelve la historia por ronda mas los pesos finales.

    `client_fraction` es la C de McMahan et al.: la fraccion de clientes que
    participa en cada ronda. Con C=1.0 participan todos, que es lo razonable
    cuando hay 3 o 5 clientes.

    `bits` activa la compresion de las actualizaciones de subida (ver
    `compresion.py`). Con None -- el default -- se promedian los pesos como
    siempre y no cambia un solo bit del comportamiento anterior, para que las
    corridas ya hechas sigan siendo reproducibles. Con un entero, cada cliente
    manda su delta cuantizado a esa cantidad de bits.

    Solo se comprime la SUBIDA. La bajada va en float32 porque el servidor
    manda el modelo global entero y, en un escenario movil, el cuello de
    botella es el ancho de banda de subida, que es el que Konecny et al.
    atacan.
    """
    rng = np.random.default_rng(seed)
    n_clients = len(particiones)
    por_ronda = max(1, int(round(client_fraction * n_clients)))

    modelo = construir_modelo()
    pesos_globales = modelo.get_weights()

    bytes_por_peso = weights_nbytes(pesos_globales)
    rng_q = np.random.default_rng(seed + 9973) if estocastico else None

    historia = {
        "ronda": [], "accuracy": [], "loss": [],
        "bytes_acumulados": [], "error_compresion": [],
    }
    bytes_subida = 0
    bytes_bajada = 0

    for ronda in range(1, rounds + 1):
        elegidos = rng.choice(n_clients, size=por_ronda, replace=False)

        updates = []
        errores = []
        for k in elegidos:
            idx = particiones[k]
            pesos_k = local_train(
                modelo,
                pesos_globales,
                X_train[idx],
                y_train_onehot[idx],
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
            )

            if bits is None:
                updates.append(([w.copy() for w in pesos_k], len(idx)))
                bytes_subida += bytes_por_peso
            else:
                delta = [wk - wg for wk, wg in zip(pesos_k, pesos_globales)]
                delta_rec, n_bytes = comprimir_update(delta, bits, rng_q)
                updates.append((delta_rec, len(idx)))
                bytes_subida += n_bytes
                errores.append(error_relativo(delta, delta_rec))

        if bits is None:
            pesos_globales = aggregate(updates)
        else:
            # Se promedian los DELTAS y se suman al global. Es identico a
            # promediar pesos porque las fracciones n_k/n suman 1.
            delta_medio = aggregate(updates)
            pesos_globales = [g + d for g, d in zip(pesos_globales, delta_medio)]

        bytes_bajada += bytes_por_peso * por_ronda
        bytes_totales = bytes_subida + bytes_bajada

        modelo.set_weights(pesos_globales)
        loss, acc = modelo.evaluate(X_test, y_test_onehot, verbose=0)

        historia["ronda"].append(ronda)
        historia["accuracy"].append(float(acc))
        historia["loss"].append(float(loss))
        historia["bytes_acumulados"].append(int(bytes_totales))
        historia["error_compresion"].append(float(np.mean(errores)) if errores else 0.0)

        if verbose:
            print(
                f"  ronda {ronda:3d}/{rounds}  acc={acc:.4f}  loss={loss:.4f}  "
                f"{bytes_totales / 1e6:7.1f} MB acumulados"
            )

    return {
        "pesos": pesos_globales,
        "historia": historia,
        "bytes_totales": int(bytes_subida + bytes_bajada),
        "bytes_subida": int(bytes_subida),
        "bytes_bajada": int(bytes_bajada),
        "modelo": modelo,
    }


def train_centralizado(
    construir_modelo,
    X_train,
    y_train_onehot,
    X_test,
    y_test_onehot,
    epochs: int,
    batch_size: int = 8,
    verbose: bool = True,
) -> dict:
    """
    Baseline centralizado: todos los datos juntos en un solo lugar.

    Es el techo teorico contra el que se compara FedAvg. No es un competidor,
    es la referencia: si el federado se le acerca, la conclusion es que se
    puede prescindir de centralizar los datos sin perder casi nada.
    """
    modelo = construir_modelo()
    modelo.fit(
        X_train,
        y_train_onehot,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1 if verbose else 0,
    )
    loss, acc = modelo.evaluate(X_test, y_test_onehot, verbose=0)
    return {"modelo": modelo, "accuracy": float(acc), "loss": float(loss), "bytes_totales": 0}
