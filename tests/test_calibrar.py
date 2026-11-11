"""Temperature scaling y métricas de calibración, sobre casos sintéticos conocidos."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from calibrar import ajustar_temperatura, cobertura, con_temperatura, ece, umbral_fijo_por_persona, umbral_sin_mirar


def test_temperatura_1_no_cambia_nada():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(con_temperatura(p, 1.0), p, atol=1e-9)


def test_temperatura_alta_aplana_y_conserva_el_orden():
    p = np.array([[0.9, 0.05, 0.05]])
    q = con_temperatura(p, 3.0)
    assert q[0, 0] < p[0, 0] and q.argmax() == 0 and abs(q.sum() - 1) < 1e-9


def test_un_modelo_sobreconfiado_pide_temperatura_mayor_a_1():
    rng = np.random.default_rng(0)
    n, c = 2000, 5
    y = rng.integers(0, c, n)
    pred = np.where(rng.random(n) < 0.5, y, rng.integers(0, c, n))  # ~60 % de acierto...
    p = np.full((n, c), 0.01)
    p[np.arange(n), pred] = 0.96  # ...con 96 % de confianza
    p /= p.sum(1, keepdims=True)
    t = ajustar_temperatura(p, y)
    assert t > 1.5
    assert ece(con_temperatura(p, t), y) < ece(p, y)


def test_ece_cero_si_la_confianza_coincide_con_el_acierto():
    p = np.array([[1.0, 0.0]] * 10)
    assert ece(p, np.zeros(10, int)) == 0.0


def test_nada_no_se_cuenta_como_mostrada():
    p = np.array([[0.9, 0.1], [0.1, 0.9]])
    fila = next(f for f in cobertura(p, np.array([0, 1]), nada=1) if abs(f["umbral"] - 0.5) < 1e-9)
    assert fila["mostradas"] == 0.5 and fila["acierto_mostradas"] == 1.0


def _caso_tres_personas():
    """Clase 0 = "nada". A y B aciertan todo con confianza 0,9 y fallan con 0,6; C acierta
    todo con 0,6 y falla con 0,9 (el caso en que mirar a todos juntos engaña)."""
    filas, y, persona = [], [], []
    for q, (conf_bien, conf_mal) in {"A": (0.9, 0.6), "B": (0.9, 0.6), "C": (0.6, 0.9)}.items():
        for _ in range(10):
            # Predice la clase 1 con conf_bien (acierta, y = 1) y con conf_mal (falla, y = 2).
            filas += [[0.0, conf_bien, 1 - conf_bien], [0.0, conf_mal, 1 - conf_mal]]
            y += [1, 2]
            persona += [q, q]
    return np.array(filas), np.array(y), np.array(persona)


def test_umbral_sin_mirar_se_elige_con_las_otras_personas():
    p, y, persona = _caso_tres_personas()
    r = umbral_sin_mirar(p, y, persona, nada=0, objetivo=0.85)
    assert set(r["umbrales"]) == {"A", "B", "C"}
    # Para C, con A y B el menor umbral que da 100 % es 0,65: en C eso muestra solo sus fallos.
    assert r["umbrales"]["C"] == 0.65
    assert r["por_persona"]["C"]["acierto_mostradas"] == 0.0 and r["por_persona"]["C"]["mostradas"] == 0.5
    assert r["acierto_min"] == 0.0


def test_umbral_fijo_por_persona_da_el_desvio_entre_personas():
    p, y, persona = _caso_tres_personas()
    r = umbral_fijo_por_persona(p, y, persona, nada=0, u=0.85)
    assert r["por_persona"]["A"] == {"mostradas": 0.5, "acierto_mostradas": 1.0}
    assert r["por_persona"]["C"] == {"mostradas": 0.5, "acierto_mostradas": 0.0}
    assert abs(r["acierto_media"] - 2 / 3) < 1e-9 and r["acierto_desvio"] > 0.5
