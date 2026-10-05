"""Export sin Flex (PY-X1) sobre la arquitectura de producción con pesos aleatorios.

No reemplaza la prueba con el modelo entrenado (scripts/probar_export_sin_flex.py
--modelo ...): con pesos aleatorios, la LSTM fusionada de TF 2.10 parecía correcta y
no lo era. Lo que sí atrapa: que vuelva a aparecer un op Flex, o que la versión
desenrollada deje de ser idéntica a la original.
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from lsa.export import convertir, desenrollar, inferir_tflite, ops_flex  # noqa: E402
from model import get_model  # noqa: E402


@pytest.fixture(scope="module")
def modelo_y_datos():
    tf.keras.utils.set_random_seed(0)
    modelo = get_model(15, 12)
    x = np.random.default_rng(0).uniform(-0.1, 1.0, (16, 15, 126)).astype(np.float32)
    return modelo, x


def test_desenrollar_no_cambia_la_matematica(modelo_y_datos):
    modelo, x = modelo_y_datos
    np.testing.assert_array_equal(modelo.predict(x, verbose=0), desenrollar(modelo).predict(x, verbose=0))


@pytest.mark.parametrize("variante", ["f32", "dinamica"])
def test_export_sin_flex(modelo_y_datos, variante):
    modelo, x = modelo_y_datos
    tfl = convertir(modelo, 15, 126, variante)
    assert ops_flex(tfl) == []
    out = inferir_tflite(tfl, x)
    tol = 1e-5 if variante == "f32" else 5e-2
    np.testing.assert_allclose(out, modelo.predict(x, verbose=0), atol=tol)
