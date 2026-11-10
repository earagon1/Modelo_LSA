"""Privacidad diferencial central (R6 de plan/federado_robustez.md): casos hechos a mano."""
import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import agregacion as ag
import protocolo as pr
import servidor as sv


def test_epsilon_a_mano():
    # R = 1, z = 1: ρ = 0,5; ε = 0,5 + 2·√(0,5·ln 10⁵).
    assert ag.epsilon_zcdp(1, 1.0) == pytest.approx(0.5 + 2 * math.sqrt(0.5 * math.log(1e5)))
    # R = 5, z = 2: ρ = 5 / 8.
    assert ag.epsilon_zcdp(5, 2.0) == pytest.approx(0.625 + 2 * math.sqrt(0.625 * math.log(1e5)))
    assert ag.epsilon_zcdp(5, 2.0) == pytest.approx(5.99, abs=0.01)
    assert ag.epsilon_zcdp(0, 1.0) == 0.0 and ag.epsilon_zcdp(3, 0.0) == math.inf
    # Más rondas gastan más; más ruido gasta menos.
    assert ag.epsilon_zcdp(10, 1.0) > ag.epsilon_zcdp(5, 1.0) > ag.epsilon_zcdp(5, 2.0)
    with pytest.raises(ValueError):
        ag.epsilon_zcdp(-1, 1.0)
    with pytest.raises(ValueError):
        ag.epsilon_zcdp(1, 1.0, delta=0)


def test_promedio_privado_sin_ruido_recorta_y_no_pondera():
    ups = [([np.array([3.0, 4.0], np.float32)], 1000), ([np.array([0.3, 0.4], np.float32)], 1)]
    r, det = ag.promedio_privado(ups, clip=1.0, ruido=0.0, rng=np.random.default_rng(0))
    # [3, 4] tiene norma 5 → [0,6; 0,8]; [0,3; 0,4] queda igual; promedio sin mirar n.
    assert r[0].tolist() == pytest.approx([0.45, 0.6])
    assert det == {"clientes": 2, "recortados": 1, "desvio_ruido": 0.0}


def test_el_recorte_es_sobre_el_update_entero():
    # Norma conjunta de los dos tensores: √(3² + 4²) = 5.
    ups = [([np.array([3.0], np.float32), np.array([4.0], np.float32)], 1)]
    r, _ = ag.promedio_privado(ups, clip=1.0, ruido=0.0, rng=np.random.default_rng(0))
    assert [x.tolist() for x in r] == [pytest.approx([0.6]), pytest.approx([0.8])]


def test_el_ruido_tiene_el_desvio_z_c_sobre_k():
    ups = [([np.zeros(200_000, np.float32)], 1) for _ in range(4)]
    r, det = ag.promedio_privado(ups, clip=2.0, ruido=3.0, rng=np.random.default_rng(1))
    assert det["desvio_ruido"] == pytest.approx(3.0 * 2.0 / 4)
    assert float(np.std(r[0])) == pytest.approx(1.5, rel=0.01) and abs(float(np.mean(r[0]))) < 0.01
    with pytest.raises(ValueError, match="recorte"):
        ag.promedio_privado(ups, clip=0.0, ruido=1.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError, match="ruido"):
        ag.promedio_privado(ups, clip=1.0, ruido=-1.0, rng=np.random.default_rng(0))


def _cabeza(c=4, semilla=0):
    rng = np.random.default_rng(semilla)
    return [rng.normal(0, 0.1, (128, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, c)).astype(np.float32), np.zeros(c, np.float32)]


def _ronda(est, p, semilla=3):
    rng = np.random.default_rng(semilla)
    deltas = []
    for i, n in enumerate((10, 500)):
        delta = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in p]
        u = {"protocolo": pr.PROTOCOLO, "ronda": est.ronda, "participant_id": f"p{i}", "n_muestras": n, "bits": 8,
             "tensores": pr.codificar_delta(delta, rng)}
        deltas.append(pr.decodificar_delta(u["tensores"]))
        est.recibir(u)
    return deltas


def test_servidor_con_dp_promedia_sin_ponderar_y_cuenta_epsilon(tmp_path):
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=2, reloj=lambda: 0.0,
                            dp_ruido=1e-9, dp_clip=5.0, dp_semilla=0)
    deltas = _ronda(est, p)
    # Con ruido despreciable y updates más chicos que C: el promedio simple, no el ponderado.
    esperado = [w + (a + b) / 2 for w, a, b in zip(p, *deltas, strict=True)]
    assert all(np.allclose(x, e, atol=1e-6) for x, e in zip(est.pesos, esperado, strict=True))
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    priv = fila["privacidad"]
    assert priv["rondas"] == 1 and priv["clip"] == 5.0 and priv["clientes"] == 2 and priv["recortados"] == 0
    assert priv["epsilon"] == ag.epsilon_zcdp(1, 1e-9)
    _ronda(est, est.pesos, semilla=4)
    assert est.estado()["privacidad"]["rondas"] == 2
    guardado = json.loads((tmp_path / "estado.json").read_text(encoding="utf-8"))
    assert guardado["rondas_dp"] == 2  # sobrevive a un reinicio (desde_paquetes lo lee)


def test_servidor_dp_con_la_misma_semilla_da_el_mismo_ruido(tmp_path):
    p = _cabeza()
    pesos = []
    for nombre in ("a", "b"):
        (tmp_path / nombre).mkdir()
        est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path / nombre), quorum=2, reloj=lambda: 0.0,
                                dp_ruido=1.0, dp_semilla=7)
        _ronda(est, p)
        pesos.append(est.pesos)
    assert all(np.array_equal(a, b) for a, b in zip(*pesos, strict=True))
    # El recorte por defecto es el límite de norma del servidor.
    assert est.dp_clip == sv.LIMITE_NORMA


def test_servidor_dp_rechaza_combinaciones_sin_sentido(tmp_path):
    with pytest.raises(ValueError, match="fedavg"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), quorum=3, agregacion="mediana",
                          dp_ruido=1.0)
    with pytest.raises(ValueError, match="recorte"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), limite_norma=None, dp_ruido=1.0)
    with pytest.raises(ValueError, match="dp_ruido"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), dp_ruido=-1.0)
    est = sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path))
    assert est.estado()["privacidad"] is None
