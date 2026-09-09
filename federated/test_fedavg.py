"""
Tests de correccion del modulo federado.

Estos tests no miden si el modelo aprende: miden si la implementacion de
FedAvg hace lo que dice hacer. Es la diferencia entre "programe algo" y "valide
una implementacion", y es lo que sostiene el capitulo de metodologia.

    1. test_promediado_de_clientes_identicos
       K clientes con pesos identicos -> el promedio es ese mismo peso.
       Si esto falla, el agregador esta deformando los pesos.

    2. test_ponderacion_por_cantidad_de_muestras
       Cliente A con 90 muestras y B con 10 -> el resultado queda 9 veces mas
       cerca de A. Si esto falla, se esta promediando sin ponderar y los
       clientes con pocos datos pesan de mas.

    3. test_identidad_un_cliente_equivale_a_centralizado
       Un solo cliente con TODOS los datos y una epoca local tiene que dar
       exactamente lo mismo que entrenar de forma centralizada una epoca.
       Es la prueba mas fuerte: si el promediado o el ciclo de rondas
       corrompieran algo, este test lo detecta.

    4. test_arquitectura_replica_el_original
       La red de federated/model.py tiene que ser identica a la de
       model.py, o la comparacion contra el modelo en produccion
       no significa nada.

Se corre en CPU a proposito: los kernels cuDNN de LSTM no son deterministas y
el test de identidad exige reproducibilidad bit a bit.

Uso:
    python federated/test_fedavg.py
    pytest federated/test_fedavg.py -v      (si tenes pytest)
"""

from __future__ import annotations

import os

# Antes de importar TF: sin GPU no hay no-determinismo de cuDNN.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fedavg import aggregate, local_train  # noqa: E402
from model import build_model  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
SEED = 1234


def _datos_sinteticos(n=64, t=15, d=126, n_classes=3, seed=SEED):
    """Datos de juguete: los tests de correccion no necesitan senas reales."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, t, d)).astype(np.float32)
    y = rng.integers(0, n_classes, size=n)
    return X, tf.keras.utils.to_categorical(y, n_classes).astype(np.float32)


# --------------------------------------------------------------------------
# 1. Promediado
# --------------------------------------------------------------------------
def test_promediado_de_clientes_identicos():
    rng = np.random.default_rng(SEED)
    pesos = [rng.normal(size=s).astype(np.float32) for s in [(4, 3), (3,), (3, 2)]]

    for k in [2, 3, 5, 7]:
        updates = [([w.copy() for w in pesos], 50) for _ in range(k)]
        resultado = aggregate(updates)
        for original, obtenido in zip(pesos, resultado):
            assert np.allclose(original, obtenido, rtol=1e-6, atol=1e-7), (
                f"Con {k} clientes identicos el promedio deberia ser el mismo peso"
            )


# --------------------------------------------------------------------------
# 2. Ponderacion
# --------------------------------------------------------------------------
def test_ponderacion_por_cantidad_de_muestras():
    ceros = [np.zeros((4, 3), dtype=np.float32)]
    unos = [np.ones((4, 3), dtype=np.float32)]

    # A aporta 90 muestras de ceros, B aporta 10 de unos -> esperado 0.1
    resultado = aggregate([(ceros, 90), (unos, 10)])
    assert np.allclose(resultado[0], 0.1, atol=1e-6), (
        f"Esperado 0.1 en todas las posiciones, se obtuvo {resultado[0].ravel()[:3]}"
    )

    # El resultado tiene que estar 9 veces mas cerca de A que de B.
    dist_a = float(np.abs(resultado[0] - ceros[0]).mean())
    dist_b = float(np.abs(resultado[0] - unos[0]).mean())
    assert np.isclose(dist_b / dist_a, 9.0, rtol=1e-5), (
        f"La razon de distancias deberia ser 9, dio {dist_b / dist_a:.4f}"
    )

    # Un promedio SIN ponderar daria 0.5: verificamos que no es lo que pasa.
    assert not np.allclose(resultado[0], 0.5), "Se esta promediando sin ponderar"


# --------------------------------------------------------------------------
# 3. Identidad
# --------------------------------------------------------------------------
def test_identidad_un_cliente_equivale_a_centralizado():
    X, y = _datos_sinteticos()
    n_classes = y.shape[1]

    tf.keras.utils.set_random_seed(SEED)
    modelo = build_model(X.shape[1], X.shape[2], n_classes)
    pesos_iniciales = [w.copy() for w in modelo.get_weights()]

    # Centralizado: una epoca sobre todos los datos.
    tf.keras.utils.set_random_seed(SEED)
    modelo.set_weights(pesos_iniciales)
    modelo.fit(X, y, epochs=1, batch_size=8, shuffle=False, verbose=0)
    pesos_centralizado = [w.copy() for w in modelo.get_weights()]

    # Federado con un unico cliente que tiene todos los datos.
    tf.keras.utils.set_random_seed(SEED)
    pesos_cliente = local_train(
        modelo, pesos_iniciales, X, y, epochs=1, batch_size=8, shuffle=False
    )
    pesos_federado = aggregate([([w.copy() for w in pesos_cliente], len(X))])

    for i, (a, b) in enumerate(zip(pesos_centralizado, pesos_federado)):
        max_dif = float(np.max(np.abs(a - b)))
        assert max_dif < 1e-6, (
            f"Tensor {i}: FedAvg con 1 cliente difiere del centralizado "
            f"(max |dif| = {max_dif:.3e})"
        )


# --------------------------------------------------------------------------
# 4. Arquitectura
# --------------------------------------------------------------------------
def _cargar_modelo_original():
    """
    Importa el model.py de produccion por ruta explicita.

    Se hace por ruta y no con `import model` porque este paquete tiene su
    propio model.py y ganaria la colision. Se inyecta un stub de `constants`
    para no arrastrar cv2 ni las rutas relativas a os.getcwd().
    """
    stub = types.ModuleType("constants")
    stub.LENGTH_KEYPOINTS = 126
    sys.modules.setdefault("constants", stub)

    ruta = RAIZ / "model.py"
    spec = importlib.util.spec_from_file_location("modelo_lsa_original", ruta)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _firma(modelo):
    """Resumen comparable de la arquitectura: tipo de capa y sus parametros clave."""
    firma = []
    for capa in modelo.layers:
        cfg = capa.get_config()
        firma.append(
            (
                type(capa).__name__,
                cfg.get("units"),
                cfg.get("rate"),
                cfg.get("activation"),
                cfg.get("return_sequences"),
            )
        )
    return firma


def test_arquitectura_replica_el_original():
    original = _cargar_modelo_original()

    tf.keras.utils.set_random_seed(SEED)
    modelo_original = original.get_model(15, 10)
    tf.keras.utils.set_random_seed(SEED)
    modelo_federado = build_model(15, 126, 10)

    assert _firma(modelo_federado) == _firma(modelo_original), (
        "federated/model.py se desincronizo de modelo_LSA/model.py.\n"
        f"  federado: {_firma(modelo_federado)}\n"
        f"  original: {_firma(modelo_original)}"
    )

    n_fed = sum(int(np.prod(w.shape)) for w in modelo_federado.get_weights())
    n_ori = sum(int(np.prod(w.shape)) for w in modelo_original.get_weights())
    assert n_fed == n_ori, f"Distinta cantidad de parametros: {n_fed} vs {n_ori}"


# --------------------------------------------------------------------------
# 5. Lectura del dataset exportado por la app
# --------------------------------------------------------------------------
def _json_de_la_app(ruta: Path, t=15, d=126):
    """
    Escribe un lsa_samples.json con el mismo esquema que produce
    `appendSamplesByDate` en CameraScreen.kt, incluidas dos muestras viejas
    sin `client_id` para cubrir la compatibilidad hacia atras.
    """
    import json as _json

    rng = np.random.default_rng(SEED)

    def muestra(label, client_id=None):
        item = {"label": label, "seq": rng.normal(size=(t, d)).round(4).tolist()}
        if client_id:
            item["client_id"] = client_id
        return item

    datos = {
        "t": t,
        "d": d,
        "by_date": {
            "2026-03-01": [
                muestra("hola", "user_aaa"),
                muestra("hola", "user_aaa"),
                muestra("gracias", "user_bbb"),
            ],
            "2026-03-02": [
                muestra("gracias", "user_bbb"),
                muestra("hola", "user_ccc"),
                muestra("chau"),  # exportada por una version vieja de la app
                muestra("chau"),
            ],
        },
    }
    ruta.write_text(_json.dumps(datos), encoding="utf-8")


def test_carga_json_con_client_id():
    import tempfile

    from data import SIN_CLIENTE, cargar_lsa_json, partition_by_client

    with tempfile.TemporaryDirectory() as tmp:
        ruta = Path(tmp) / "lsa_samples.json"
        _json_de_la_app(ruta)

        X, y, etiquetas, client_ids = cargar_lsa_json(
            ruta, etiquetas_base=["hola", "gracias"], verbose=False
        )

        assert X.shape == (7, 15, 126), f"shape inesperado: {X.shape}"
        assert etiquetas == ["hola", "gracias", "chau"], (
            f"El orden de words.json debe respetarse y las nuevas ir al final: {etiquetas}"
        )

        # Las muestras sin client_id no se pierden: quedan identificables.
        assert (client_ids == SIN_CLIENTE).sum() == 2

        partes = partition_by_client(client_ids)
        assert len(partes) == 4, f"Esperaba 4 grupos (3 personas + sin_cliente), hay {len(partes)}"
        assert sum(len(p) for p in partes) == len(X), "La particion debe cubrir todas las muestras"

        # Cada grupo tiene un unico client_id.
        for parte in partes:
            assert len(set(client_ids[parte])) == 1

        # user_aaa grabo dos 'hola'; el indice de 'hola' es 0.
        grupo_aaa = next(p for p in partes if client_ids[p][0] == "user_aaa")
        assert len(grupo_aaa) == 2 and set(y[grupo_aaa]) == {0}


# --------------------------------------------------------------------------
# 6-8. Compresion de las actualizaciones
# --------------------------------------------------------------------------
def test_deltas_equivalen_a_promediar_pesos():
    """
    Sin cuantizar, mandar deltas tiene que dar lo mismo que mandar pesos.

    Es la base algebraica de todo el esquema de compresion: si esto no valiera,
    cualquier diferencia que apareciera despues seria del cambio de
    representacion y no de la cuantizacion, y el experimento no mediria nada.
    """
    from fedavg import aggregate

    rng = np.random.default_rng(SEED)
    formas = [(6, 4), (4,), (4, 3)]
    globales = [rng.normal(size=f).astype(np.float32) for f in formas]
    clientes = [
        [g + rng.normal(scale=0.1, size=g.shape).astype(np.float32) for g in globales]
        for _ in range(3)
    ]
    tamanos = [90, 10, 50]

    via_pesos = aggregate(list(zip(clientes, tamanos)))

    deltas = [[c - g for c, g in zip(cli, globales)] for cli in clientes]
    delta_medio = aggregate(list(zip(deltas, tamanos)))
    via_deltas = [g + d for g, d in zip(globales, delta_medio)]

    for i, (a, b) in enumerate(zip(via_pesos, via_deltas)):
        assert np.allclose(a, b, rtol=1e-5, atol=1e-6), (
            f"Tensor {i}: promediar deltas difiere de promediar pesos "
            f"(max |dif| = {np.max(np.abs(a - b)):.3e})"
        )


def test_cuantizacion_determinista_acota_el_error():
    """Con redondeo a lo mas cercano el error nunca supera media escala."""
    from compresion import cuantizar, descuantizar

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(200, 30)).astype(np.float32)

    for bits in [8, 4, 2]:
        q, escala = cuantizar(x, bits, rng=None)
        recuperado = descuantizar(q, escala)
        error_max = float(np.max(np.abs(x - recuperado)))
        assert error_max <= escala / 2 + 1e-6, (
            f"{bits} bits: error {error_max:.5f} supera media escala {escala / 2:.5f}"
        )


def test_cuantizacion_estocastica_es_insesgada():
    """
    El redondeo estocastico tiene que tener media cero de error.

    Es la propiedad por la que Konecny et al. lo eligen: el sesgo del redondeo
    determinista no se cancela al promediar entre clientes y se acumula ronda
    tras ronda, mientras que el ruido insesgado si se cancela.
    """
    from compresion import cuantizar, descuantizar

    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(500,)).astype(np.float32)

    # Promediar muchas realizaciones deberia converger al original.
    acumulado = np.zeros_like(x, dtype=np.float64)
    repeticiones = 400
    for _ in range(repeticiones):
        q, escala = cuantizar(x, bits=4, rng=rng)
        acumulado += descuantizar(q, escala)
    promedio = acumulado / repeticiones

    sesgo_estocastico = float(np.mean(np.abs(promedio - x)))

    q_det, escala_det = cuantizar(x, bits=4, rng=None)
    sesgo_determinista = float(np.mean(np.abs(descuantizar(q_det, escala_det) - x)))

    assert sesgo_estocastico < sesgo_determinista / 3, (
        f"El estocastico deberia promediar mucho mejor que el determinista: "
        f"{sesgo_estocastico:.6f} contra {sesgo_determinista:.6f}"
    )


def test_bytes_bajan_con_los_bits():
    """El ahorro tiene que ser proporcional a los bits, sin sorpresas."""
    from compresion import bytes_de_update

    tensores = [np.zeros((1000, 64), dtype=np.float32), np.zeros((64,), dtype=np.float32)]
    b32 = bytes_de_update(tensores, 32)
    b8 = bytes_de_update(tensores, 8)
    b4 = bytes_de_update(tensores, 4)

    assert b32 == 64 * 1064 // 16 * 4 or b32 == (1000 * 64 + 64) * 4
    assert abs(b8 / b32 - 0.25) < 0.01, f"8 bits deberia ser ~1/4 de float32, dio {b8 / b32:.3f}"
    assert abs(b4 / b32 - 0.125) < 0.01, f"4 bits deberia ser ~1/8, dio {b4 / b32:.3f}"


# --------------------------------------------------------------------------
if __name__ == "__main__":
    pruebas = [
        test_promediado_de_clientes_identicos,
        test_ponderacion_por_cantidad_de_muestras,
        test_identidad_un_cliente_equivale_a_centralizado,
        test_arquitectura_replica_el_original,
        test_carga_json_con_client_id,
        test_deltas_equivalen_a_promediar_pesos,
        test_cuantizacion_determinista_acota_el_error,
        test_cuantizacion_estocastica_es_insesgada,
        test_bytes_bajan_con_los_bits,
    ]

    print("=" * 68)
    print("TESTS DE CORRECCION DEL MODULO FEDERADO")
    print("=" * 68)

    fallaron = 0
    for prueba in pruebas:
        try:
            prueba()
            print(f"[ OK ]  {prueba.__name__}")
        except AssertionError as e:
            fallaron += 1
            print(f"[FALLA] {prueba.__name__}\n        {e}")
        except Exception as e:  # error de codigo, no de asercion
            fallaron += 1
            print(f"[ERROR] {prueba.__name__}\n        {type(e).__name__}: {e}")

    print("=" * 68)
    print(f"{len(pruebas) - fallaron}/{len(pruebas)} tests pasaron")
    sys.exit(1 if fallaron else 0)
