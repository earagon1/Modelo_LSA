"""Protocolo y servidor del federado por red local (PY-F6, contrato federado-v1)."""
import json
import os
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import protocolo as pr
import servidor as sv


def _cabeza(c=4, semilla=0):
    rng = np.random.default_rng(semilla)
    return [rng.normal(0, 0.1, (128, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, c)).astype(np.float32), np.zeros(c, np.float32)]


def _update(pesos, ronda, pid, n=10, escala=0.01, semilla=1):
    rng = np.random.default_rng(semilla)
    delta = [rng.normal(0, escala, w.shape).astype(np.float32) for w in pesos]
    return {"protocolo": pr.PROTOCOLO, "ronda": ronda, "participant_id": pid, "n_muestras": n, "bits": 8,
            "tensores": pr.codificar_delta(delta, rng)}, delta


def test_pesos_ida_y_vuelta_y_sha():
    p = _cabeza()
    q = pr.decodificar_pesos(json.loads(json.dumps(pr.codificar_pesos(p))))
    assert all(np.array_equal(a, b) for a, b in zip(p, q, strict=True))
    assert pr.sha_pesos(p) == pr.sha_pesos(q)
    q[0][0, 0] += 1e-7
    assert pr.sha_pesos(p) != pr.sha_pesos(q)


def test_cuantizacion_con_uniformes_es_determinista_e_insesgada():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000).astype(np.float32)
    u = rng.random(5000, dtype=np.float32)
    q1, e1 = pr.cuantizar(x, u)
    q2, e2 = pr.cuantizar(x, u)
    assert np.array_equal(q1, q2) and e1 == e2 and q1.dtype == np.int8
    assert np.abs(q1).max() <= 127
    # Insesgado: promediando muchos redondeos de un mismo valor se recupera el valor.
    y = np.full(20000, 0.3, np.float32)
    y[0] = 1.0  # fija la escala en 1/127
    q, e = pr.cuantizar(y, rng.random(20000, dtype=np.float32))
    assert abs(pr.descuantizar(q, e)[1:].mean() - 0.3) < 2e-4


def test_agregar_es_fedavg_ponderado():
    w = [np.zeros(3, np.float32)]
    nueva = pr.agregar(w, [([np.ones(3, np.float32)], 30), ([np.full(3, -1, np.float32)], 10)])
    assert np.allclose(nueva[0], 0.5)


def test_rotacion_ida_y_vuelta_y_conserva_la_norma():
    rng = np.random.default_rng(0)
    for n in (1, 13, 64, 832, 8192):
        x = rng.normal(0, 1, n).astype(np.float32)
        y = pr.rotar(x, 99)
        assert len(y) == pr.potencia_de_2(n)
        assert abs(np.linalg.norm(y) - np.linalg.norm(x)) < 1e-3 * max(1.0, np.linalg.norm(x))
        assert np.abs(pr.desrotar(y, 99, n) - x).max() < 1e-5
    s = pr.signos(5, 20000)
    assert set(np.unique(s)) == {-1.0, 1.0} and abs(s.mean()) < 0.03


def test_delta_rotado_se_reconstruye_y_paga_el_relleno():
    p = _cabeza(c=13)
    rng = np.random.default_rng(1)
    delta = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in p]
    t = pr.codificar_delta(delta, rng, semilla=40)
    rec = pr.decodificar_delta(json.loads(json.dumps(t)))
    err = pr.norma([a - b for a, b in zip(delta, rec, strict=True)]) / pr.norma(delta)
    assert err < 0.03
    assert pr.validar_update({"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": "p1_a", "n_muestras": 5,
                              "bits": 8, "tensores": t}, 0, [list(w.shape) for w in p]) is None
    # La capa de salida (64×13 = 832 valores) viaja completada a 1024.
    assert pr.bytes_payload(t) - pr.bytes_payload(pr.codificar_delta(delta, rng)) == (1024 - 832) + (16 - 13) + 8 * 6


def test_el_limite_recorta_solo_los_updates_grandes(tmp_path):
    p = _cabeza()
    rng = np.random.default_rng(2)
    chico = [rng.normal(0, 1e-3, w.shape).astype(np.float32) for w in p]
    d, n, rec = pr.recortar(chico, 2.0)
    assert not rec and d is chico
    grande = [x * np.float32(5000) for x in chico]
    d, n, rec = pr.recortar(grande, 2.0)
    assert rec and abs(pr.norma(d) - 2.0) < 1e-3 and n > 2.0
    # En el servidor: el update malicioso entra recortado y queda registrado.
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=1, reloj=Reloj(), limite_norma=2.0)
    u = {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": "p1_malo", "n_muestras": 10, "bits": 8,
         "tensores": pr.codificar_delta(grande, rng)}
    r = est.recibir(u)[1]
    assert r["recortado"]
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert fila["recortados"] == ["p1_malo"] and fila["normas"]["p1_malo"] > 2.0
    assert pr.norma([a - b for a, b in zip(est.pesos, p, strict=True)]) <= 2.0 + 1e-3


def test_validar_rechaza_lo_que_no_corresponde():
    p = _cabeza()
    formas = [list(w.shape) for w in p]
    u, _ = _update(p, 0, "p1_x")
    assert pr.validar_update(u, 0, formas) is None
    assert "ronda" in pr.validar_update(u, 1, formas)
    assert "formas" in pr.validar_update(u, 0, formas[:-1])
    assert "n_muestras" in pr.validar_update({**u, "n_muestras": 0}, 0, formas)
    assert "bits" in pr.validar_update({**u, "bits": 4}, 0, formas)


class Reloj:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_ronda_por_quorum_publica_y_es_idempotente(tmp_path):
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=2, reloj=Reloj())
    u1, _ = _update(p, 0, "p1_a", n=30, semilla=1)
    assert est.recibir(u1)[1]["aceptado"]
    assert est.recibir(u1)[1]["motivo"].startswith("duplicado")
    u2, _ = _update(p, 0, "p1_b", n=10, semilla=2)
    r = est.recibir(u2)[1]
    assert r["cierre"]["publicada"] and est.version == 1 and est.ronda == 1
    esperado = pr.agregar(p, [(pr.decodificar_delta(u1["tensores"]), 30), (pr.decodificar_delta(u2["tensores"]), 10)])
    assert all(np.array_equal(a, b) for a, b in zip(est.pesos, esperado, strict=True))
    assert est.recibir(u1)[0] == 409  # ronda vieja
    fila = json.loads((tmp_path / "rondas.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert fila["clientes"] == ["p1_a", "p1_b"] and fila["bytes_subidos"]["p1_a"] == sum(w.size + 4 for w in p)
    assert (tmp_path / "cabezas" / "v1.json").exists()


def test_timeout_cierra_con_un_solo_cliente(tmp_path):
    reloj = Reloj()
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=3, timeout_s=60, reloj=reloj)
    assert est.revisar_timeout() is None
    est.recibir(_update(p, 0, "p1_a")[0])
    reloj.t = 59
    assert est.revisar_timeout() is None
    reloj.t = 61
    assert est.revisar_timeout()["ronda_cerrada"] == 0 and est.ronda == 1


def test_el_plazo_corre_desde_el_primer_update(tmp_path):
    # Una ronda abierta horas sin nadie no la cierra el primero que llega.
    reloj = Reloj()
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=2, timeout_s=60, reloj=reloj)
    reloj.t = 10_000
    est.recibir(_update(p, 0, "p1_a")[0])
    assert est.revisar_timeout() is None and est.ronda == 0
    assert est.estado()["segundos_restantes"] == 60
    reloj.t = 10_061
    assert est.revisar_timeout()["ronda_cerrada"] == 0


def test_el_control_de_calidad_no_publica_una_cabeza_peor(tmp_path):
    p = _cabeza(c=4)
    rng = np.random.default_rng(3)
    emb = rng.normal(0, 1, (200, 128)).astype(np.float32)
    y = pr.predecir(p, emb).argmax(1)  # la cabeza actual acierta el 100 %
    est = sv.EstadoServidor(p, list("abcd"), "sha", str(tmp_path), quorum=1, test=(emb, y), reloj=Reloj())
    malo, _ = _update(p, 0, "p1_malo", escala=0.5)  # un delta enorme y al azar
    cierre = est.recibir(malo)[1]["cierre"]
    assert not cierre["publicada"] and est.version == 0 and est.ronda == 1
    assert all(np.array_equal(a, b) for a, b in zip(est.pesos, p, strict=True))


def test_http_de_punta_a_punta(tmp_path):
    p = _cabeza()
    est = sv.EstadoServidor(p, list("abcd"), "sha-bb", str(tmp_path), quorum=1, reloj=Reloj())
    srv = sv.crear_servidor(est, host="127.0.0.1", puerto=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        m = json.loads(urllib.request.urlopen(base + "/modelo").read())
        pesos = pr.decodificar_pesos(m["pesos"])
        assert m["sha256"] == pr.sha_pesos(pesos) and m["backbone_sha256"] == "sha-bb" and m["version"] == 0
        u, _ = _update(pesos, m["ronda"], "p1_a")
        req = urllib.request.Request(base + "/update", json.dumps(u).encode(), {"Content-Type": "application/json"})
        r = json.loads(urllib.request.urlopen(req).read())
        assert r["aceptado"] and r["cierre"]["version"] == 1
        assert json.loads(urllib.request.urlopen(base + "/estado").read())["ronda"] == 1
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(urllib.request.Request(base + "/update", b"{no json", {}))
        assert e.value.code == 400
    finally:
        srv.shutdown()


def _toma(uuid, label="hola", pid="p1_a"):
    return {"esquema": "senar-muestra-v2", "uuid": uuid, "label": label, "participant_id": pid,
            "frames": [{"t_ms": 0, "manos": [{"e": "Right", "s": 0.9, "lm": [0.5] * 63}]}] * 10}


def test_muestras_aportadas_validadas_y_sin_repetir(tmp_path):
    est = sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path), reloj=Reloj())
    cuerpo = {"protocolo": pr.PROTOCOLO, "participant_id": "p1_a",
              "muestras": [_toma("u1"), _toma("u2", "chau"), _toma("u3", "Cómo"), _toma("u4", pid="p1_otro")]}
    codigo, r = est.recibir_muestras(cuerpo)
    assert codigo == 200 and r["aceptadas"] == 2 and len(r["rechazadas"]) == 2
    r = est.recibir_muestras({**cuerpo, "muestras": [_toma("u1"), _toma("u5")]})[1]
    assert r["aceptadas"] == 1 and r["repetidas"] == 1 and r["total_aportadas"] == 3
    lineas = (tmp_path / "muestras" / "p1_a.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(x)["uuid"] for x in lineas] == ["u1", "u2", "u5"]
    # Al reiniciar el servidor recuerda lo que ya tenía.
    assert sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path)).uuids_aportados == {"u1", "u2", "u5"}
    assert est.recibir_muestras({**cuerpo, "participant_id": "../x"})[0] == 400


def _publicar_paquete(directorio, n, pesos, clases):
    carpeta = directorio / "paquetes" / f"v{n}"
    carpeta.mkdir(parents=True)
    capas = [{"W": pesos[i].tolist(), "b": pesos[i + 1].tolist()} for i in range(0, len(pesos), 2)]
    (carpeta / "cabeza.json").write_text(json.dumps({"clases": clases, "capas": capas}), encoding="utf-8")
    (carpeta / "backbone.tflite").write_bytes(b"bb" + bytes([n]))
    archivos = [{"nombre": a, "sha256": "x", "bytes": (carpeta / a).stat().st_size}
                for a in ("backbone.tflite", "cabeza.json")]
    (carpeta / "manifiesto.json").write_text(json.dumps(
        {"protocolo": pr.PROTOCOLO, "paquete": n, "clases": clases, "archivos": archivos,
         "backbone_sha256": f"bb{n}"}), encoding="utf-8")


def test_paquete_nuevo_reemplaza_la_cabeza_y_descarta_lo_pendiente(tmp_path):
    p = _cabeza()
    _publicar_paquete(tmp_path, 1, p, list("abcd"))
    est = sv.EstadoServidor.desde_paquetes(str(tmp_path), quorum=3, reloj=Reloj())
    assert est.paquete == 1 and est.modelo()["backbone_sha256"] == "bb1"
    est.recibir(_update(p, 0, "p1_a")[0])
    # Paquete 2 con una seña nueva: la cabeza tiene una salida más.
    p2 = _cabeza(c=5, semilla=9)
    _publicar_paquete(tmp_path, 2, p2, list("abcde"))
    evento = est.revisar_paquete()
    assert evento["updates_descartados"] == ["p1_a"] and est.pendientes == {}
    m = est.modelo()
    assert m["paquete"] == 2 and m["clases"] == list("abcde") and m["arquitectura"]["capas"] == [64, 64, 5]
    assert est.revisar_paquete() is None
    # Un update armado sobre la cabeza vieja ya no encaja.
    assert est.recibir(_update(p, 0, "p1_b")[0])[0] == 400
    assert est.archivo_de_paquete(2, "backbone.tflite").endswith("backbone.tflite")
    assert est.archivo_de_paquete(2, "../../estado.json") is None


def test_al_reiniciar_sigue_la_ronda_y_la_cabeza(tmp_path):
    p = _cabeza()
    _publicar_paquete(tmp_path, 1, p, list("abcd"))
    est = sv.EstadoServidor.desde_paquetes(str(tmp_path), quorum=1, reloj=Reloj())
    est.recibir(_update(p, 0, "p1_a")[0])
    est.recibir(_update(est.pesos, 1, "p1_a", semilla=5)[0])
    otra = sv.EstadoServidor.desde_paquetes(str(tmp_path), quorum=1, reloj=Reloj())
    assert (otra.ronda, otra.version, otra.paquete) == (est.ronda, est.version, 1) == (2, 3, 1)
    assert pr.sha_pesos(otra.pesos) == pr.sha_pesos(est.pesos) != pr.sha_pesos(p)
    assert otra.manifiesto["paquete"] == 1


def test_http_reparte_el_paquete(tmp_path):
    _publicar_paquete(tmp_path, 1, _cabeza(), list("abcd"))
    est = sv.EstadoServidor.desde_paquetes(str(tmp_path), quorum=1, reloj=Reloj())
    srv = sv.crear_servidor(est, host="127.0.0.1", puerto=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        man = json.loads(urllib.request.urlopen(base + "/paquete").read())
        assert man["paquete"] == 1
        assert urllib.request.urlopen(base + "/paquete/v1/backbone.tflite").read() == b"bb\x01"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/paquete/v1/manifiesto2.json")
        assert e.value.code == 404
        cuerpo = json.dumps({"protocolo": pr.PROTOCOLO, "participant_id": "p1_a", "muestras": [_toma("u9")]})
        r = json.loads(urllib.request.urlopen(urllib.request.Request(base + "/muestras", cuerpo.encode())).read())
        assert r["aceptadas"] == 1
    finally:
        srv.shutdown()
