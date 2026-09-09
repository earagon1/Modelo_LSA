"""
Carga y particionado del dataset de senas para los experimentos federados.

El dataset vive en `data/keypoints/*.h5`: un archivo por sena, cada
uno con una columna `sample` que agrupa los frames de una misma ejecucion y una
columna `keypoints` con el vector de 126 valores de ese frame.

El particionado es la pieza que convierte ese conjunto unico en N clientes, que
es lo que FedAvg necesita para existir. Hay tres formas de partir:

  - `partition_iid`:       reparto uniforme al azar. Es el caso facil y sirve
                           como referencia optimista.
  - `partition_dirichlet`: reparto desbalanceado por clase, controlado por alpha.
                           Es el caso realista (non-IID): cada persona senia un
                           subconjunto distinto del vocabulario y con distinta
                           frecuencia. alpha chico = mas desbalance.
  - `partition_by_client`: cuando el dataset traiga de quien es cada muestra.
                           Es el que hay que usar en cuanto haya grabaciones de
                           varias personas; los otros dos son sustitutos.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Raiz del repo Modelo_LSA, resuelta desde este archivo para que los scripts
# corran igual desde cualquier directorio.
RAIZ = Path(__file__).resolve().parent.parent
KEYPOINTS_DIR = RAIZ / "data" / "keypoints"
WORDS_JSON = RAIZ / "models" / "words.json"

T_FRAMES = 15
D_FEATURES = 126


def cargar_etiquetas(words_json: Path = WORDS_JSON) -> list[str]:
    """Lee el orden de clases de words.json. Ese orden define los indices."""
    with open(words_json, "r", encoding="utf-8") as f:
        return json.load(f)["word_ids"]


def _normalizar_largo(seq: np.ndarray, t: int = T_FRAMES) -> np.ndarray:
    """
    Lleva una secuencia a exactamente `t` frames.

    Se replica el criterio de `training_model.py` (padding 'pre', truncating
    'post') para que los resultados sean comparables con el modelo que ya esta
    en produccion: si sobran frames se descartan los del final, si faltan se
    rellena con ceros al principio.
    """
    n = len(seq)
    if n == t:
        return seq
    if n > t:
        return seq[:t]
    relleno = np.zeros((t - n, seq.shape[1]), dtype=seq.dtype)
    return np.concatenate([relleno, seq], axis=0)


def cargar_dataset(
    keypoints_dir: Path = KEYPOINTS_DIR,
    words_json: Path = WORDS_JSON,
    t: int = T_FRAMES,
    d: int = D_FEATURES,
    conservar_indices_de_words_json: bool = False,
    verbose: bool = True,
):
    """
    Devuelve (X, y, etiquetas) leyendo un .h5 por clase.

    X: (N, t, d) float32 -- se usa float32 y no el float16 de `training_model.py`
       porque aca hay que promediar pesos entre clientes y no vale la pena
       arrastrar error de cuantizacion en la entrada.
    y: (N,) int64 con el indice de clase segun el orden de words.json.

    Las clases de words.json que no tengan .h5 se ignoran, igual que hace
    `training_model.py`, y se avisa por consola.

    `conservar_indices_de_words_json` cambia como se numeran las clases:

      - False (default): los indices son 0..K-1 sobre las clases PRESENTES.
        Es lo correcto para entrenar un modelo nuevo, como hace el experimento
        federado, porque no tiene sentido reservar una salida para una clase
        sin datos.

      - True: los indices son los de words.json completo, huecos incluidos.
        Es obligatorio para EVALUAR un modelo ya entrenado: su capa de salida
        se numero con el words.json de entonces, y si aca se renumeran las
        clases presentes, todas las etiquetas posteriores a un hueco quedan
        corridas y la accuracy sale mal medida sin que nada falle.
    """
    etiquetas_todas = cargar_etiquetas(words_json)

    presentes, faltantes = [], []
    for etiqueta in etiquetas_todas:
        if (keypoints_dir / f"{etiqueta}.h5").exists():
            presentes.append(etiqueta)
        else:
            faltantes.append(etiqueta)

    if faltantes and verbose:
        print(f"[data] Sin .h5, se ignoran: {faltantes}")

    if not presentes:
        raise FileNotFoundError(f"No se encontro ningun .h5 en {keypoints_dir}")

    secuencias, indices = [], []
    for idx, etiqueta in enumerate(presentes):
        tabla = pd.read_hdf(keypoints_dir / f"{etiqueta}.h5", key="data")
        n_muestras = 0
        for _, grupo in tabla.groupby("sample"):
            seq = np.asarray(
                [np.asarray(fila, dtype=np.float32) for fila in grupo["keypoints"]],
                dtype=np.float32,
            )
            if seq.ndim != 2 or seq.shape[1] != d:
                continue
            secuencias.append(_normalizar_largo(seq, t))
            indices.append(idx)
            n_muestras += 1
        if verbose:
            print(f"[data] {etiqueta:15s} {n_muestras:4d} muestras")

    X = np.stack(secuencias, axis=0).astype(np.float32)
    y = np.asarray(indices, dtype=np.int64)

    if conservar_indices_de_words_json and faltantes:
        # Se remapea de la numeracion de presentes a la de words.json completo.
        remapeo = {i: etiquetas_todas.index(e) for i, e in enumerate(presentes)}
        y = np.asarray([remapeo[i] for i in y], dtype=np.int64)
        salida = etiquetas_todas
    else:
        salida = presentes

    if verbose:
        print(f"[data] Total: {len(X)} muestras, {len(salida)} clases, shape={X.shape}")

    return X, y, salida


SIN_CLIENTE = "sin_cliente"


def cargar_lsa_json(
    json_path: Path,
    t: int = T_FRAMES,
    d: int = D_FEATURES,
    etiquetas_base: list[str] | None = None,
    verbose: bool = True,
):
    """
    Lee el `lsa_samples.json` que exporta la app y devuelve (X, y, etiquetas, client_ids).

    Esquema del archivo:

        {"t": 15, "d": 126,
         "by_date": {"2026-03-01": [{"label": "hola",
                                     "client_id": "user_2abc...",
                                     "seq": [[...126...], ...]}]}}

    `client_id` es el id de la cuenta que grabo la muestra. Es el campo que
    habilita `partition_by_client`, o sea el experimento federado con clientes
    reales en vez de particiones sinteticas. Las muestras exportadas por
    versiones anteriores de la app no lo traen y se agrupan bajo `SIN_CLIENTE`;
    el conteo se avisa por consola porque esas muestras no sirven para el
    experimento por persona.

    Si se pasa `etiquetas_base` (tipicamente el `word_ids` de words.json), se
    respeta ese orden y las etiquetas nuevas se agregan al final, igual que
    hace `training_model.py`.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        datos = json.load(f)

    t = int(datos.get("t", t))
    d = int(datos.get("d", d))

    secuencias, etiquetas_txt, clientes = [], [], []
    for _, items in (datos.get("by_date") or {}).items():
        for item in items:
            seq = np.asarray(item["seq"], dtype=np.float32)
            if seq.ndim != 2 or seq.shape[1] != d:
                continue
            secuencias.append(_normalizar_largo(seq, t))
            etiquetas_txt.append(item["label"])
            clientes.append(item.get("client_id") or SIN_CLIENTE)

    if not secuencias:
        raise ValueError(f"{json_path} no tiene ninguna muestra utilizable")

    etiquetas = list(etiquetas_base or [])
    for lab in sorted(set(etiquetas_txt)):
        if lab not in etiquetas:
            etiquetas.append(lab)
    # Solo se conservan las clases que realmente aparecen en el archivo.
    etiquetas = [e for e in etiquetas if e in set(etiquetas_txt)]

    indice = {lab: i for i, lab in enumerate(etiquetas)}
    X = np.stack(secuencias, axis=0).astype(np.float32)
    y = np.asarray([indice[lab] for lab in etiquetas_txt], dtype=np.int64)
    client_ids = np.asarray(clientes)

    if verbose:
        print(f"[data] {json_path.name}: {len(X)} muestras, {len(etiquetas)} clases")
        for cid in sorted(set(clientes)):
            n = int((client_ids == cid).sum())
            marca = "  <-- sin atribucion" if cid == SIN_CLIENTE else ""
            print(f"[data]   {cid:24s} {n:4d} muestras{marca}")
        if SIN_CLIENTE in set(clientes):
            print(
                "[data] AVISO: hay muestras sin client_id (exportadas por una version "
                "de la app anterior al campo). No sirven para particionar por persona."
            )

    return X, y, etiquetas, client_ids


def split_train_test(X, y, test_size: float = 0.2, seed: int = 42):
    """
    Split estratificado. El test es global y ningun cliente lo ve nunca: es la
    unica forma de comparar centralizado contra federado en igualdad.
    """
    from sklearn.model_selection import train_test_split

    return train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )


def partition_iid(y, n_clients: int, seed: int = 42) -> list[np.ndarray]:
    """Reparto uniforme al azar. Cada cliente ve la misma distribucion de clases."""
    rng = np.random.default_rng(seed)
    orden = rng.permutation(len(y))
    return [np.sort(p) for p in np.array_split(orden, n_clients)]


def partition_dirichlet(
    y,
    n_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
    min_por_cliente: int = 10,
    max_intentos: int = 100,
) -> list[np.ndarray]:
    """
    Reparto non-IID: para cada clase se sortea un vector de proporciones
    p ~ Dir(alpha) y se reparten las muestras de esa clase segun p.

    alpha -> 0   : cada cliente acapara pocas clases (muy non-IID).
    alpha -> inf : tiende al reparto IID.

    Es el procedimiento estandar en la literatura de federated learning para
    fabricar heterogeneidad reproducible cuando no hay particion natural.

    Se reintenta hasta que ningun cliente quede por debajo de `min_por_cliente`,
    porque un cliente casi vacio rompe el entrenamiento local sin aportar nada.
    """
    rng = np.random.default_rng(seed)
    clases = np.unique(y)

    for intento in range(max_intentos):
        partes = [[] for _ in range(n_clients)]
        for c in clases:
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            props = rng.dirichlet(np.repeat(alpha, n_clients))
            cortes = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cliente, trozo in enumerate(np.split(idx_c, cortes)):
                partes[cliente].extend(trozo.tolist())

        tamanos = [len(p) for p in partes]
        if min(tamanos) >= min_por_cliente:
            return [np.sort(np.asarray(p, dtype=np.int64)) for p in partes]

    raise RuntimeError(
        f"No se logro una particion con >= {min_por_cliente} muestras por cliente "
        f"tras {max_intentos} intentos (alpha={alpha}, n_clients={n_clients}). "
        f"Proba subir alpha o bajar n_clients."
    )


def partition_by_client(client_ids, seed: int = 42) -> list[np.ndarray]:
    """
    Particion natural: un cliente por persona que grabo.

    Es la que hay que usar en cuanto el dataset guarde de quien es cada muestra.
    Hasta entonces las otras dos son sustitutos declarados como tales.
    """
    client_ids = np.asarray(client_ids)
    return [np.where(client_ids == cid)[0] for cid in sorted(set(client_ids.tolist()))]


def reporte_particion(partes: list[np.ndarray], y, etiquetas: list[str]) -> str:
    """Tabla de cuantas muestras de cada clase le toco a cada cliente."""
    lineas = []
    encabezado = "cliente".ljust(10) + "total".rjust(7) + "  " + "".join(
        e[:8].rjust(9) for e in etiquetas
    )
    lineas.append(encabezado)
    lineas.append("-" * len(encabezado))
    for i, parte in enumerate(partes):
        conteo = np.bincount(y[parte], minlength=len(etiquetas))
        fila = f"c{i}".ljust(10) + str(len(parte)).rjust(7) + "  "
        fila += "".join(str(int(n)).rjust(9) for n in conteo)
        lineas.append(fila)
    return "\n".join(lineas)
