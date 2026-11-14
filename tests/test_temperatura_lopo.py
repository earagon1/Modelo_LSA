"""Las formas de elegir la temperatura que compara scripts/temperatura_lopo.py (ML-M7)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from temperatura_lopo import aplicar_por_pliegue, comparar, metricas, temperatura_anidada, temperatura_oraculo


def _caso():
    """Dos semillas × tres personas, tres filas cada una. La clase 0 es "nada"."""
    p = np.array([[0.05, 0.9, 0.05], [0.1, 0.2, 0.7], [0.6, 0.3, 0.1]] * 6)
    y = np.array([1, 1, 0] * 6)
    persona = np.repeat(np.array(["A", "B", "C"] * 2), 3)
    semilla = np.repeat(np.array([0, 1]), 9)
    return p, y, persona, semilla


def test_la_anidada_no_mira_a_la_persona_ni_a_otra_semilla():
    p, y, persona, semilla = _caso()
    vistos = []

    def ajustar(pp, yy):
        vistos.append(len(yy))
        return float(len(yy))
    t = temperatura_anidada(p, y, persona, semilla, ajustar=ajustar)
    assert set(t) == {(s, q) for s in (0, 1) for q in "ABC"}
    # Cada pliegue se ajusta con las 6 filas de las otras dos personas de su semilla.
    assert vistos == [6] * 6


def test_la_anidada_usa_exactamente_las_filas_de_las_otras():
    p, y, persona, semilla = _caso()
    p = p.copy()
    p[(persona == "A") & (semilla == 0)] = [[1 / 3] * 3] * 3  # marca las filas de A, semilla 0
    usadas = []
    temperatura_anidada(p, y, persona, semilla, ajustar=lambda pp, yy: usadas.append(pp.copy()) or 1.0)
    # El pliegue (0, A) es el primero y no puede ver las filas marcadas; (0, B) sí las ve.
    assert not np.any(np.all(np.isclose(usadas[0], 1 / 3), axis=1))
    assert np.any(np.all(np.isclose(usadas[1], 1 / 3), axis=1))


def test_el_oraculo_mira_solo_a_la_persona():
    p, y, persona, semilla = _caso()
    t = temperatura_oraculo(p, y, persona, semilla, ajustar=lambda pp, yy: float(len(yy)))
    assert all(v == 3.0 for v in t.values())


def test_aplicar_con_t_1_no_cambia_y_por_pliegue_aplica_la_suya():
    p, _, persona, semilla = _caso()
    np.testing.assert_allclose(aplicar_por_pliegue(p, persona, semilla, 1.0), p, atol=1e-9)
    temps = {(s, q): (3.0 if q == "B" else 1.0) for s in (0, 1) for q in "ABC"}
    q = aplicar_por_pliegue(p, persona, semilla, temps)
    np.testing.assert_allclose(q[persona == "A"], p[persona == "A"], atol=1e-9)
    assert (q[persona == "B"].max(1) < p[persona == "B"].max(1)).all()  # T > 1 aplana
    assert (q.argmax(1) == p.argmax(1)).all()  # y no cambia la predicción


def test_metricas_con_el_umbral_cuenta_nada_como_no_mostrada():
    p = np.array([[0.05, 0.9, 0.05], [0.1, 0.2, 0.7], [0.8, 0.1, 0.1], [0.1, 0.5, 0.4]])
    y = np.array([1, 1, 0, 1])
    m = metricas(p, y, nada=0, umbral=0.6)
    # Mostradas: la 1 (0,9, acierta) y la 2 (0,7, falla); la 3 es "nada" y la 4 no llega.
    assert m["n_mostradas"] == 2 and m["mostradas"] == 0.5 and m["acierto_mostradas"] == 0.5
    assert m["accuracy"] == 0.75
    assert metricas(p, y, nada=0, umbral=0.99)["acierto_mostradas"] is None


def test_comparar_da_las_mismas_personas_en_cada_esquema():
    p, y, persona, semilla = _caso()
    r = comparar(p, y, persona, semilla, nada=0, umbral=0.6,
                 esquemas={"sin": 1.0, "fija": 2.0, "anidada": temperatura_anidada(p, y, persona, semilla)})
    for e in ("sin", "fija", "anidada"):
        assert set(r[e]["metricas"]) == {"A", "B", "C", "todas"}
        assert r[e]["metricas"]["todas"]["n"] == len(y)
    # T no cambia la argmax: el accuracy es el mismo con cualquier esquema.
    assert r["sin"]["metricas"]["todas"]["accuracy"] == r["fija"]["metricas"]["todas"]["accuracy"]
    # Más T, menos confianza: con el mismo umbral se muestra lo mismo o menos.
    assert r["fija"]["metricas"]["todas"]["mostradas"] <= r["sin"]["metricas"]["todas"]["mostradas"]
    assert set(r["anidada"]["temperaturas"]) == {f"{s}-{q}" for s in (0, 1) for q in "ABC"}
