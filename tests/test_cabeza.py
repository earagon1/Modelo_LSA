"""FedAvg de la cabeza (PY-F4): el backbone no se mueve y solo viaja la cabeza."""
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")
# Al final del path: así "model" sigue siendo el model.py de la raíz para los demás tests.
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import cabeza  # noqa: E402

build_model = cabeza.build_model


@pytest.fixture(scope="module")
def datos():
    tf.keras.utils.set_random_seed(0)
    rng = np.random.default_rng(0)
    base = build_model(15, 126, 5).get_weights()
    clientes = []
    for _ in range(2):
        X = rng.normal(size=(20, 15, 126)).astype(np.float32)
        y = tf.keras.utils.to_categorical(rng.integers(0, 5, 20), 5)
        clientes.append((X, y))
    return base, clientes


def test_solo_la_cabeza_es_entrenable(datos):
    base, _ = datos
    m = cabeza.construir(base, 15, 126, 5)
    # 128·64+64 + 64·64+64 + 64·5+5 = 12 581
    assert cabeza.n_parametros_cabeza(m) == 128 * 64 + 64 + 64 * 64 + 64 + 64 * 5 + 5


def test_el_backbone_no_cambia_bit_a_bit(datos):
    base, clientes = datos
    m = cabeza.construir(base, 15, 126, 5)
    lstm_antes = [w.copy() for c in m.layers if isinstance(c, tf.keras.layers.LSTM) for w in c.get_weights()]
    final, hist = cabeza.fedavg(m, clientes, clientes[0][0], clientes[0][1], rondas=2, epocas_locales=1)
    lstm_despues = [w for c in m.layers if isinstance(c, tf.keras.layers.LSTM) for w in c.get_weights()]
    for a, b in zip(lstm_antes, lstm_despues, strict=True):
        np.testing.assert_array_equal(a, b)
    assert any(not np.array_equal(a, b) for a, b in zip(final, cabeza.cabeza(cabeza.construir(base, 15, 126, 5)),
                                                        strict=True)), "la cabeza sí tiene que cambiar"
    assert hist["bytes_subida"][-1] == 2 * 2 * cabeza.n_parametros_cabeza(m) * 4


def test_con_8_bits_se_sube_menos(datos):
    base, clientes = datos
    m = cabeza.construir(base, 15, 126, 5)
    _, h32 = cabeza.fedavg(m, clientes, clientes[0][0], clientes[0][1], rondas=1, epocas_locales=1)
    m = cabeza.construir(base, 15, 126, 5)
    _, h8 = cabeza.fedavg(m, clientes, clientes[0][0], clientes[0][1], rondas=1, epocas_locales=1, bits=8)
    assert h8["bytes_subida"][-1] < h32["bytes_subida"][-1] / 3


def test_agregacion_robusta_sobre_la_cabeza(datos, monkeypatch):
    """El cableado de `agregacion` en cabeza.fedavg: cada cliente devuelve global + un
    delta fijo (se reemplaza el entrenamiento), así el resultado se calcula a mano."""
    base, clientes = datos
    m = cabeza.construir(base, 15, 126, 5)
    g0 = cabeza.cabeza(m)
    escalas = [0.01, 0.02, 5.0]  # el tercero es el atípico
    tres = [clientes[0], clientes[1], (clientes[0][0][:5], clientes[0][1][:5])]
    llamadas = []

    def falso(modelo, pesos_cabeza, X, y1h, epocas, batch_size=8):
        k = len(llamadas) % 3
        llamadas.append(k)
        return [w + np.float32(escalas[k]) for w in pesos_cabeza]

    monkeypatch.setattr(cabeza, "entrenar_local", falso)
    final, _ = cabeza.fedavg(m, tres, clientes[0][0], clientes[0][1], rondas=1, agregacion="mediana")
    for a, b in zip(final, g0, strict=True):
        np.testing.assert_allclose(a, b + 0.02, atol=1e-6)
    # FedAvg pondera por n (20, 20, 5 de 45): (20·0,01 + 20·0,02 + 5·5) / 45.
    m = cabeza.construir(base, 15, 126, 5)
    final, _ = cabeza.fedavg(m, tres, clientes[0][0], clientes[0][1], rondas=1)
    for a, b in zip(final, g0, strict=True):
        np.testing.assert_allclose(a, b + (20 * 0.01 + 20 * 0.02 + 5 * 5.0) / 45, atol=1e-5)
    # Con 8 bits la mediana se calcula sobre los deltas cuantizados: queda cerca de 0,02.
    m = cabeza.construir(base, 15, 126, 5)
    final, _ = cabeza.fedavg(m, tres, clientes[0][0], clientes[0][1], rondas=1, bits=8, agregacion="mediana")
    for a, b in zip(final, g0, strict=True):
        np.testing.assert_allclose(a, b + 0.02, atol=1e-3)
    with pytest.raises(ValueError, match="agregación"):
        cabeza.fedavg(m, tres, clientes[0][0], clientes[0][1], rondas=1, agregacion="bulyan")
