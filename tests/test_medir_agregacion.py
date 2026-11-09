"""Las partes puras del experimento de ataques (federated/medir_agregacion_robusta.py)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import medir_agregacion_robusta as mar
import protocolo as pr
import servidor as sv


def _cabeza(c=4, semilla=0):
    rng = np.random.default_rng(semilla)
    return [rng.normal(0, 0.1, (128, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, c)).astype(np.float32), np.zeros(c, np.float32)]


def _honestos(k=3, semilla=1):
    rng = np.random.default_rng(semilla)
    p = _cabeza()
    base = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in p]
    return p, [([b + rng.normal(0, 2e-3, b.shape).astype(np.float32) for b in base], 80 + 10 * i)
               for i in range(k)]


def test_los_ataques_tienen_la_forma_y_la_norma_que_dicen():
    _, h = _honestos()
    rng = np.random.default_rng(0)
    mediana = float(np.median([pr.norma(d) for d, _ in h]))
    medio = [np.mean([d[i] for d, _ in h], axis=0) for i in range(len(h[0][0]))]
    g, n = mar.delta_atacante("grande", h, rng)
    assert [x.shape for x in g] == [x.shape for x in h[0][0]] and n == 90
    assert pr.norma(g) == pytest.approx(50 * mediana, rel=1e-5)
    r, n = mar.delta_atacante("ruido_x2", h, rng)
    assert pr.norma(r) == pytest.approx(2 * mediana, rel=1e-5) and n == 90
    inv, n = mar.delta_atacante("invertido", h, rng)
    assert n == 90 and all(np.allclose(a, -10 * b, atol=1e-6) for a, b in zip(inv, medio, strict=True))
    inf, n = mar.delta_atacante("n_inflado", h, rng)
    assert n == 10 ** 6 and all(np.array_equal(a, b) for a, b in zip(inf, inv, strict=True))
    with pytest.raises(ValueError, match="ataque"):
        mar.delta_atacante("ninguno", h, rng)


def test_beta_descarta_al_menos_uno_por_punta():
    assert mar.beta_para(4) == 0.25 and mar.beta_para(16) == 0.2
    for k in (3, 4, 5, 16, 18):
        assert int(np.floor(mar.beta_para(k) * k)) >= 1


@pytest.mark.parametrize("nombre,quorum", [("fedavg_limite_tope", 4), ("fedavg_limite_tope200", 4),
                                          ("mediana", 4), ("krum", 5)])
def test_combinar_ronda_publica_lo_mismo_que_el_servidor(tmp_path, nombre, quorum):
    """El experimento no reimplementa el servidor por su cuenta: con los mismos deltas
    (ya pasados por los 8 bits), la cabeza que sale es la que publicaría servidor.py."""
    p, h = _honestos(quorum - 1)
    rng = np.random.default_rng(2)
    malo = mar.delta_atacante("n_inflado", h, rng)
    est_cfg = mar.ESTRATEGIAS[nombre]
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=quorum, reloj=lambda: 0.0,
                            limite_norma=est_cfg["limite"], max_muestras=est_cfg["tope"] or 0,
                            agregacion=est_cfg["agregacion"], krum_f=1, krum_m=1)
    decodificados = []
    for i, (delta, n) in enumerate([*h, malo]):
        u = {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": f"p{i}", "n_muestras": n, "bits": 8,
             "tensores": pr.codificar_delta(delta, rng)}
        decodificados.append((pr.decodificar_delta(u["tensores"]), n))
        est.recibir(u)
    assert est.version == 1
    mio, motivo = mar.combinar_ronda(p, decodificados, est_cfg, f=1, m=1)
    assert motivo is None
    assert all(np.allclose(a, b, atol=1e-6) for a, b in zip(mio, est.pesos, strict=True))
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert fila["agregacion"] == est_cfg["agregacion"]


def test_krum_sin_clientes_suficientes_no_cambia_la_cabeza():
    p, h = _honestos(3)
    nuevo, motivo = mar.combinar_ronda(p, h, mar.ESTRATEGIAS["krum"], f=1, m=1)
    assert "al menos 5" in motivo
    assert all(np.array_equal(a, b) for a, b in zip(nuevo, p, strict=True))


def test_evaluar_reporta_promedio_y_desvio_entre_personas():
    p = _cabeza(c=2)
    emb = np.zeros((4, 128), np.float32)
    y = pr.predecir(p, emb).argmax(1)  # todas aciertan...
    y[3] = 1 - y[3]  # ...menos una de P2
    r = mar.evaluar(p, emb, y, np.array(["P1", "P1", "P2", "P2"]))
    assert r["acierto"] == 0.75 and r["por_persona"] == {"P1": 1.0, "P2": 0.5}
    assert r["media_personas"] == 0.75 and r["desvio_personas"] == 0.25
