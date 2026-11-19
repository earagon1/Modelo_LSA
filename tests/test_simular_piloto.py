"""Simulación del piloto: los clientes salen de las mismas reglas que la app."""
import os
import sys
from types import SimpleNamespace

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "federated"))

from simular_piloto import tomas_de_cliente  # noqa: E402


def test_cliente_es_su_telefono_sin_el_test_fijo():
    d = SimpleNamespace(
        fuente=np.array(["telefono", "telefono", "webcam", "telefono", "telefono"]),
        persona=np.array(["P2", "P2", "P2", "P1", "P2"]),
        uuid=np.array(["a", "b", "c", "d", "e"]),
    )
    m = tomas_de_cliente(d, "P2", {"b"})
    # "b" es del test fijo, "c" es de la webcam y "d" es de otra persona: entrenan solo "a" y "e".
    assert d.uuid[m].tolist() == ["a", "e"]
