"""Agregación robusta (R2 de plan/federado_robustez.md): casos hechos a mano."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import agregacion as ag
import protocolo as pr
import servidor as sv


def _u(*valores, n=1):
    """Un update de un solo tensor 1-D con esos valores."""
    return ([np.array(valores, np.float32)], n)


def test_mediana_a_mano():
    r = ag.mediana([_u(1, 10), _u(2, 20), _u(100, -50)])
    assert r[0].tolist() == [2, 10]
    # k par: promedio de los dos del medio.
    r = ag.mediana([_u(1), _u(2), _u(3), _u(100)])
    assert r[0].tolist() == [2.5]


def test_mediana_ignora_n():
    # Si pesara n, el 1000 con n = 10⁶ ganaría.
    r = ag.mediana([_u(1, n=1), _u(2, n=1), _u(1000, n=10**6)])
    assert r[0].tolist() == [2]


def test_media_recortada_a_mano():
    # k = 5, beta = 0,2 -> saca 1 de cada punta: queda (2 + 3 + 4) / 3.
    ups = [_u(v) for v in (3, 100, 2, -50, 4)]
    assert ag.media_recortada(ups, 0.2)[0].tolist() == [3]
    # beta = 0: media sin ponderar.
    assert ag.media_recortada(ups, 0.0)[0].tolist() == pytest.approx([(3 + 100 + 2 - 50 + 4) / 5])
    # beta·k < 1 (k = 3, beta = 0,2): no saca nada, aunque n sea distinto.
    assert ag.media_recortada([_u(0, n=1), _u(3, n=100), _u(6, n=1)], 0.2)[0].tolist() == [3]


def test_media_recortada_con_k3_y_beta_un_tercio_es_la_mediana():
    ups = [_u(1, 7), _u(5, -2), _u(900, 3)]
    assert ag.media_recortada(ups, 0.34)[0].tolist() == ag.mediana(ups)[0].tolist()


def test_updates_iguales_devuelven_el_mismo():
    t = [np.arange(6, dtype=np.float32).reshape(2, 3), np.ones(4, np.float32)]
    ups = [([x.copy() for x in t], n) for n in (1, 5, 9)]
    for r in (ag.mediana(ups), ag.media_recortada(ups, 0.2)):
        assert all(np.array_equal(a, b) and a.dtype == np.float32 for a, b in zip(r, t, strict=True))


def test_por_coordenada_y_varios_tensores():
    # Cada coordenada se resuelve sola: el atípico puede ser de un cliente distinto en cada una.
    ups = [([np.array([[0, 9]], np.float32), np.array([5], np.float32)], 1),
           ([np.array([[1, 1]], np.float32), np.array([-99], np.float32)], 1),
           ([np.array([[50, 2]], np.float32), np.array([6], np.float32)], 1)]
    r = ag.mediana(ups)
    assert r[0].tolist() == [[1, 2]] and r[1].tolist() == [5]


def test_entradas_invalidas():
    with pytest.raises(ValueError, match="no hay"):
        ag.mediana([])
    with pytest.raises(ValueError, match="formas"):
        ag.mediana([_u(1, 2), _u(1, 2, 3)])
    with pytest.raises(ValueError, match="cantidad"):
        ag.mediana([([np.zeros(1)], 1), ([np.zeros(1), np.zeros(1)], 1)])
    for beta in (-0.1, 0.5, 0.7):
        with pytest.raises(ValueError, match="beta"):
            ag.media_recortada([_u(1)], beta)
    with pytest.raises(ValueError, match="estrategia"):
        ag.combinar([_u(1)], "fedavg")


def test_advertencias():
    assert "quórum" in ag.advertencia("mediana", 2)
    assert ag.advertencia("mediana", 3) is None
    assert "5 clientes" in ag.advertencia("recortada", 4, 0.2)
    assert ag.advertencia("recortada", 5, 0.2) is None


def test_agregar_fedavg_no_cambia_y_las_robustas_suman_al_peso():
    w = [np.array([10, 20], np.float32)]
    ups = [([np.array([1, 1], np.float32)], 1), ([np.array([2, 2], np.float32)], 1),
           ([np.array([30, -30], np.float32)], 98)]
    assert pr.agregar(w, ups)[0].tolist() == pytest.approx([10 + 0.01 + 0.02 + 29.4, 20 + 0.01 + 0.02 - 29.4])
    assert pr.agregar(w, ups, "fedavg")[0].tolist() == pr.agregar(w, ups)[0].tolist()
    assert pr.agregar(w, ups, "mediana")[0].tolist() == [12, 21]


def _cabeza(c=4, semilla=0):
    rng = np.random.default_rng(semilla)
    return [rng.normal(0, 0.1, (128, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, c)).astype(np.float32), np.zeros(c, np.float32)]


class _Reloj:
    def __call__(self):
        return 0.0


def test_servidor_con_mediana_resiste_al_atacante(tmp_path):
    """Dos honestos parecidos y un atacante con un delta enorme y n inflado. Sin límite
    de norma ni tope de n, para ver solo el efecto de la estrategia."""
    p = _cabeza()
    rng = np.random.default_rng(3)
    base = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in p]
    honestos = [[b + rng.normal(0, 1e-3, b.shape).astype(np.float32) for b in base] for _ in range(2)]
    malo = [np.float32(-500) * b for b in base]

    def ronda(estrategia):
        d = tmp_path / estrategia
        d.mkdir()
        est = sv.EstadoServidor(p, list("abcd"), "sha", str(d), quorum=3, reloj=_Reloj(),
                                limite_norma=None, max_muestras=None, agregacion=estrategia)
        for i, (delta, n) in enumerate([(honestos[0], 20), (honestos[1], 30), (malo, 10**6)]):
            u = {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": f"p{i}", "n_muestras": n, "bits": 8,
                 "tensores": pr.codificar_delta(delta, rng)}
            assert est.recibir(u)[1]["aceptado"]
        fila = json.loads((d / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        movido = [a - b for a, b in zip(est.pesos, p, strict=True)]
        return pr.norma([m - b for m, b in zip(movido, base, strict=True)]) / pr.norma(base), fila

    error_fedavg, fila = ronda("fedavg")
    assert error_fedavg > 400 and fila["agregacion"] == "fedavg" and fila["advertencia_agregacion"] is None
    error_mediana, fila = ronda("mediana")
    # La mediana queda entre los dos honestos (más el error de los 8 bits).
    assert error_mediana < 0.2 and fila["agregacion"] == "mediana" and fila["advertencia_agregacion"] is None


def test_servidor_rechaza_estrategia_o_beta_invalidos(tmp_path):
    with pytest.raises(ValueError, match="agregación"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), agregacion="bulyan")
    with pytest.raises(ValueError, match="beta"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), agregacion="recortada", beta=0.5)


def test_ronda_por_tiempo_con_pocos_clientes_queda_advertida(tmp_path):
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=3, timeout_s=10, reloj=_Reloj(),
                            agregacion="mediana")
    rng = np.random.default_rng(0)
    u = {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": "p0", "n_muestras": 5, "bits": 8,
         "tensores": pr.codificar_delta([np.zeros_like(w) for w in p], rng)}
    est.recibir(u)
    est.timeout_s = 0  # el reloj no avanza: se fuerza el cierre por tiempo
    assert est.revisar_timeout() is not None
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert fila["motivo"] == "timeout" and "1 cliente" in fila["advertencia_agregacion"]
    assert est.estado()["agregacion"] == "mediana"


# --- Krum (R3) ---

def test_puntajes_y_eleccion_de_krum_a_mano():
    """k = 5, f = 1: cada uno suma las distancias² a sus 5 − 1 − 2 = 2 vecinos más cercanos.
    Puntos 0, 1, 2, 3 y 100: puntajes 1+4, 1+1, 1+1, 1+4 y 97²+98²."""
    ups = [_u(v) for v in (0, 1, 2, 3, 100)]
    assert ag.puntajes_krum(ups, 1).tolist() == [5, 2, 2, 5, 97 ** 2 + 98 ** 2]
    # Empate entre 1 y 2: gana el que llegó primero.
    r, elegidos = ag.krum(ups, f=1)
    assert elegidos == [1] and r[0].tolist() == [1]
    # Multi-Krum: promedio sin ponderar de los m mejores.
    assert ag.krum(ups, f=1, m=2)[0][0].tolist() == [1.5]
    r, elegidos = ag.krum(ups, f=1, m=3)
    assert elegidos == [1, 2, 0] and r[0].tolist() == [1]


def test_krum_elige_el_update_entero_y_no_mira_n():
    # Dos coordenadas: Krum devuelve un update que existe, no una mezcla por coordenada.
    ups = [_u(0, 0), _u(1, 1), _u(0, 1), _u(1, 0), _u(50, 50, n=10**6)]
    r, elegidos = ag.krum(ups, f=1)
    assert 4 not in elegidos and r[0].tolist() in ([0, 0], [1, 1], [0, 1], [1, 0])


def test_distancias_de_krum_coinciden_con_la_cuenta_directa():
    rng = np.random.default_rng(7)
    ups = [([rng.normal(size=(3, 4)).astype(np.float32), rng.normal(size=5).astype(np.float32)], 1)
           for _ in range(7)]
    planos = [np.concatenate([np.asarray(t, np.float64).ravel() for t in u]) for u, _ in ups]
    f, vecinos = 2, 7 - 2 - 2
    directo = [sorted(float(np.sum((a - b) ** 2)) for j, b in enumerate(planos) if j != i)[:vecinos]
               for i, a in enumerate(planos)]
    assert ag.puntajes_krum(ups, f) == pytest.approx([sum(d) for d in directo], rel=1e-9)


def test_krum_exige_suficientes_clientes():
    assert ag.clientes_minimos_krum(0) == 3 and ag.clientes_minimos_krum(1) == 5
    with pytest.raises(ValueError, match="al menos 5"):
        ag.krum([_u(v) for v in range(4)], f=1)
    with pytest.raises(ValueError, match="m=5"):
        ag.krum([_u(v) for v in range(5)], f=1, m=5)
    for f in (-1, 1.0, True):
        with pytest.raises(ValueError, match="f "):
            ag.clientes_minimos_krum(f)
    with pytest.raises(ValueError, match="m "):
        ag.validar_krum(5, 1, 0)
    assert "al menos 5" in ag.advertencia("krum", 3, f=1) and ag.advertencia("krum", 5, f=1) is None


def _update_de(p, delta, pid, n, rng):
    return {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": pid, "n_muestras": n, "bits": 8,
            "tensores": pr.codificar_delta(delta, rng)}


def test_servidor_no_arranca_krum_con_quorum_chico(tmp_path):
    with pytest.raises(ValueError, match="quórum 3"):
        sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), quorum=3, agregacion="krum", krum_f=1)


def test_servidor_con_krum_deja_afuera_al_atacante(tmp_path):
    p = _cabeza()
    rng = np.random.default_rng(4)
    base = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in p]
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=5, reloj=_Reloj(),
                            limite_norma=None, max_muestras=None, agregacion="krum", krum_f=1, krum_m=3)
    for i in range(4):
        honesto = [b + rng.normal(0, 1e-3, b.shape).astype(np.float32) for b in base]
        est.recibir(_update_de(p, honesto, f"p{i}", 20, rng))
    r = est.recibir(_update_de(p, [np.float32(-500) * b for b in base], "p_malo", 10**6, rng))[1]
    assert r["cierre"]["publicada"]
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert fila["krum"]["f"] == 1 and fila["krum"]["m"] == 3 and len(fila["krum"]["elegidos"]) == 3
    assert "p_malo" not in fila["krum"]["elegidos"] and fila["advertencia_agregacion"] is None
    movido = [a - b for a, b in zip(est.pesos, p, strict=True)]
    assert pr.norma([m - b for m, b in zip(movido, base, strict=True)]) / pr.norma(base) < 0.2


def test_ronda_krum_por_tiempo_con_pocos_clientes_no_publica(tmp_path):
    p = _cabeza()
    rng = np.random.default_rng(5)
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=5, timeout_s=10, reloj=_Reloj(),
                            agregacion="krum", krum_f=1)
    for i in range(3):
        est.recibir(_update_de(p, [np.full_like(w, 0.01) for w in p], f"p{i}", 10, rng))
    est.timeout_s = 0
    cierre = est.revisar_timeout()
    assert cierre["publicada"] is False and est.version == 0 and est.ronda == 1
    assert all(np.array_equal(a, b) for a, b in zip(est.pesos, p, strict=True))
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "al menos 5" in fila["advertencia_agregacion"] and fila["krum"]["elegidos"] is None
    assert fila["accuracy_candidata"] is None
