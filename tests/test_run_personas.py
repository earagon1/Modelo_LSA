"""El resumen del federado con personas: la prueba t va sobre personas, no sobre semillas."""
import os
import sys

import pytest
from scipy.stats import ttest_rel

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "federated"))

from run_personas import resumir  # noqa: E402


def _fila_a(persona, semilla):
    return {"persona": persona, "semilla": semilla, "base": 0.5, "fedavg": 0.6, "fedavg_8b": 0.6, "techo": 0.7,
            "piso": 0.55, "rondas_95_techo": 3, "bytes_fedavg": 100, "bytes_fedavg_8b": 25}


def _filas(mejoras, semillas):
    """Cada persona mejora siempre lo mismo: repetir semillas no agrega evidencia."""
    return [{"persona": p, "semilla": s, "global": 0.6 + 0.001 * s, "personalizado": 0.6 + 0.001 * s + m}
            for s in range(semillas) for p, m in mejoras.items()]


def test_la_prueba_usa_una_observacion_por_persona():
    mejoras = {"P1": 0.07, "P2": 0.01, "P3": 0.04}
    a = [_fila_a(p, s) for s in range(5) for p in mejoras]
    r = resumir(a, _filas(mejoras, 5), _filas(mejoras, 5), ttest_rel)["usuario_nuevo"]
    esperado = ttest_rel([0.67, 0.61, 0.64], [0.60, 0.60, 0.60])
    assert r["n_prueba"] == 3
    assert r["p"] == pytest.approx(esperado.pvalue, rel=1e-6)
    assert r["mejora_por_persona"] == pytest.approx(mejoras)


def test_repetir_semillas_no_cambia_el_valor_p():
    mejoras = {"P1": 0.07, "P2": 0.01, "P3": 0.04}
    p = [resumir([_fila_a(q, s) for s in range(n) for q in mejoras], _filas(mejoras, n), _filas(mejoras, n),
                 ttest_rel)["miembros"]["p"] for n in (1, 5)]
    assert p[0] == pytest.approx(p[1], rel=1e-6)


def test_con_una_sola_persona_no_hay_prueba():
    r = resumir([_fila_a("P1", 0)], _filas({"P1": 0.05}, 2), _filas({"P1": 0.05}, 2), ttest_rel)["usuario_nuevo"]
    assert "p" not in r and r["mejora_por_persona"] == pytest.approx({"P1": 0.05})
