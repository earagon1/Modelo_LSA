"""TLS y códigos de participación del servidor federado (R7 y R8)."""
import base64
import hashlib
import json
import os
import shutil
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import generar_certificado as gen
import protocolo as pr
import seguridad as seg
import servidor as sv

OPENSSL = shutil.which("openssl")
necesita_openssl = pytest.mark.skipif(OPENSSL is None, reason="sin openssl en el PATH")


def _cabeza(c=4):
    rng = np.random.default_rng(0)
    return [rng.normal(0, 0.1, (128, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, 64)).astype(np.float32), np.zeros(64, np.float32),
            rng.normal(0, 0.1, (64, c)).astype(np.float32), np.zeros(c, np.float32)]


# --- códigos de participación ---

def test_alta_verificar_y_revocar(tmp_path):
    aut = seg.AutenticadorCodigos(str(tmp_path / "participantes.json"))
    codigo = aut.alta("p_3f2a91c0")
    assert len(codigo) == 16 and set(codigo) <= set(seg.ALFABETO)
    assert aut.verificar(f"Bearer {codigo}") == "p_3f2a91c0"
    # Como lo escribe una persona: con guiones, espacios y minúsculas.
    assert aut.verificar(f"Bearer {seg.codigo_para_mostrar(codigo).lower().replace('-', ' - ')}") == "p_3f2a91c0"
    for mala in (None, "", codigo, f"Basic {codigo}", "Bearer ", "Bearer AAAAAAAAAAAAAAAA"):
        assert aut.verificar(mala) is None, mala
    # El archivo no guarda el código, solo su sha256.
    guardado = (tmp_path / "participantes.json").read_text(encoding="utf-8")
    assert codigo not in guardado and hashlib.sha256(codigo.encode()).hexdigest() in guardado
    # Un alta nueva reemplaza al anterior.
    otro = aut.alta("p_3f2a91c0")
    assert aut.verificar(f"Bearer {codigo}") is None and aut.verificar(f"Bearer {otro}") == "p_3f2a91c0"
    assert aut.revocar("p_3f2a91c0") and not aut.revocar("p_3f2a91c0")
    assert aut.verificar(f"Bearer {otro}") is None and aut.participantes() == []
    with pytest.raises(ValueError):
        aut.alta("p 1/../x")


def test_codigo_para_mostrar():
    assert seg.codigo_para_mostrar("abcdEFGHjkmnPQRS") == "ABCD-EFGH-JKMN-PQRS"
    assert seg.normalizar_codigo(" abcd-efgh ") == "ABCDEFGH"


def _servidor(tmp_path, tls=None, autenticador=None):
    est = sv.EstadoServidor(_cabeza(), list("abcd"), "sha", str(tmp_path / "srv"), quorum=1, reloj=lambda: 0.0)
    srv = sv.crear_servidor(est, host="127.0.0.1", puerto=0, tls=tls, autenticador=autenticador)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return est, srv


def _post(url, cuerpo, codigo=None, contexto=None):
    cab = {"Content-Type": "application/json"}
    if codigo:
        cab["Authorization"] = f"Bearer {codigo}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, json.dumps(cuerpo).encode(), cab),
                                    context=contexto, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_con_codigo(tmp_path):
    aut = seg.AutenticadorCodigos(str(tmp_path / "participantes.json"))
    codigo = aut.alta("p1_a")
    est, srv = _servidor(tmp_path, autenticador=aut)
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        # Los GET son públicos.
        m = json.loads(urllib.request.urlopen(base + "/modelo", timeout=10).read())
        pesos = pr.decodificar_pesos(m["pesos"])
        rng = np.random.default_rng(1)
        delta = [rng.normal(0, 0.01, w.shape).astype(np.float32) for w in pesos]
        u = {"protocolo": pr.PROTOCOLO, "ronda": 0, "participant_id": "p1_a", "n_muestras": 10, "bits": 8,
             "tensores": pr.codificar_delta(delta, rng)}
        assert _post(base + "/update", u)[0] == 401
        assert _post(base + "/update", u, "AAAA-AAAA-AAAA-AAAA")[0] == 401
        # Con un código válido pero el cuerpo a nombre de otro: 403, y no entra.
        codigo_otro = aut.alta("p2_b")
        c, r = _post(base + "/update", u, codigo_otro)
        assert c == 403 and "participant_id" in r["motivo"] and est.ronda == 0
        c, r = _post(base + "/update", u, codigo)
        assert c == 200 and r["aceptado"] and est.ronda == 1
        c, _ = _post(base + "/muestras", {"protocolo": pr.PROTOCOLO, "participant_id": "p1_a", "muestras": []},
                     codigo_otro)
        assert c == 403
    finally:
        srv.shutdown()


# --- TLS ---

@necesita_openssl
def test_huella_igual_que_openssl(tmp_path):
    cert, h = gen.generar(str(tmp_path))
    pub = subprocess.run([OPENSSL, "x509", "-in", cert, "-pubkey", "-noout"], check=True,
                         capture_output=True).stdout
    der = subprocess.run([OPENSSL, "pkey", "-pubin", "-outform", "der"], input=pub, check=True,
                         capture_output=True).stdout
    assert h == hashlib.sha256(der).digest()
    assert len(seg.huella_para_mostrar(h).split()) == 5
    assert base64.b64decode(seg.huella_base64(h)) == h
    with pytest.raises(SystemExit, match="--forzar"):
        gen.generar(str(tmp_path))
    _, h2 = gen.generar(str(tmp_path), forzar=True)
    assert h2 != h  # clave nueva, huella nueva


@necesita_openssl
def test_https_de_punta_a_punta(tmp_path):
    cert, _ = gen.generar(str(tmp_path / "srv"))
    clave = os.path.join(os.path.dirname(cert), "servidor.key")
    _, srv = _servidor(tmp_path, tls=seg.contexto_tls(cert, clave))
    base = f"https://127.0.0.1:{srv.server_address[1]}"
    try:
        confia = ssl.create_default_context(cafile=cert)
        confia.check_hostname = False
        assert json.loads(urllib.request.urlopen(base + "/estado", context=confia, timeout=10).read())["ronda"] == 0
        # Sin confiar en ese certificado, el handshake falla: no hay HTTP en claro detrás.
        with pytest.raises(urllib.error.URLError):
            urllib.request.urlopen(base + "/estado", context=ssl.create_default_context(), timeout=10)
        with pytest.raises((urllib.error.URLError, ConnectionError)):
            urllib.request.urlopen(base.replace("https", "http") + "/estado", timeout=10)
        # Y después de un cliente rechazado, el servidor sigue atendiendo.
        assert json.loads(urllib.request.urlopen(base + "/estado", context=confia, timeout=10).read())["ronda"] == 0
    finally:
        srv.shutdown()


def test_muestra_estructural():
    base = {"esquema": "senar-muestra-v2", "uuid": "u1", "label": "hola", "participant_id": "p1",
            "img_w": 640, "img_h": 480,
            "frames": [{"t_ms": 0, "manos": [{"e": "Right", "s": 0.9, "lm": [0.5] * 63}]}]}
    assert sv.validar_muestra(base, "p1") is None
    mano = base["frames"][0]["manos"][0]
    casos = {
        "img_w": {**base, "img_w": 0}, "img_h": {**base, "img_h": "480"},
        "objeto": {**base, "frames": ["no"]},
        "dos manos": {**base, "frames": [{"manos": [mano, mano, mano]}]},
        "lateralidad": {**base, "frames": [{"manos": [{**mano, "e": "Centro"}]}]},
        "score": {**base, "frames": [{"manos": [{**mano, "s": 1.5}]}]},
        "21 landmarks": {**base, "frames": [{"manos": [{**mano, "lm": [0.5] * 62}]}]},
        "finitos": {**base, "frames": [{"manos": [{**mano, "lm": [0.5] * 62 + [float("nan")]}]}]},
        "números": {**base, "frames": [{"manos": [{**mano, "lm": [0.5] * 62 + ["x"]}]}]},
    }
    for esperado, m in casos.items():
        motivo = sv.validar_muestra(m, "p1")
        assert motivo is not None and esperado.split()[0] in motivo, (esperado, motivo)
