"""El servidor todavía reproduce el fixture que la app usa como referencia.

contract/fixtures/paridad_federado_v1.json lo genera scripts/generar_fixture_federado.py
con federated/protocolo.py, y ProtocoloFederadoTest (en TraductorLSA) comprueba que
Kotlin da lo mismo bit a bit. Faltaba el lado de Python (auditoría, punto 12): si
alguien cambiaba protocolo.py sin regenerar el fixture, la app seguía pasando contra
un fixture viejo y el servidor decodificaba otra cosa que lo que la app mandaba.
"""
import json
import os
import sys

import numpy as np
import pytest

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(RAIZ, "federated"))
import protocolo as pr  # noqa: E402

FIXTURE = os.path.join(RAIZ, "contract", "fixtures", "paridad_federado_v1.json")


def _fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _f32(valores, forma=None):
    x = np.asarray(valores, dtype=np.float32)
    return x.reshape(forma) if forma is not None else x


def test_es_el_protocolo_del_servidor():
    assert _fixture()["protocolo"] == pr.PROTOCOLO


@pytest.mark.parametrize("caso", _fixture()["cuantizacion"], ids=lambda c: c["nombre"])
def test_cuantizacion_bit_a_bit(caso):
    x = _f32(caso["x"], caso["forma"])
    q, escala = pr.cuantizar(x, _f32(caso["uniformes"], caso["forma"]))
    assert q.ravel().tolist() == caso["q"]
    # La escala viaja en float32: tiene que ser exactamente la misma, no aproximada.
    assert np.float32(escala) == np.float32(caso["escala"])
    assert pr.a_b64(q, np.int8) == caso["q_base64"]
    assert np.array_equal(pr.descuantizar(q, escala).ravel(), _f32(caso["reconstruido"]))


@pytest.mark.parametrize("caso", _fixture()["cuantizacion"], ids=lambda c: c["nombre"])
def test_el_servidor_decodifica_lo_que_manda_la_app(caso):
    # Un tensor de POST /update armado con los valores del fixture, como lo arma la app.
    tensor = {"forma": caso["forma"], "escala": caso["escala"], "datos": caso["q_base64"]}
    (delta,) = pr.decodificar_delta([tensor])
    assert delta.shape == tuple(caso["forma"])
    assert np.array_equal(delta.ravel(), _f32(caso["reconstruido"]))


def test_sha_y_base64_de_los_pesos():
    s = _fixture()["sha_pesos"]
    pesos = [_f32(t["datos"], t["forma"]) for t in s["tensores"]]
    assert pr.sha_pesos(pesos) == s["sha256"]
    assert [pr.a_b64(w, np.float32) for w in pesos] == s["base64"]
    vuelta = pr.decodificar_pesos(
        [{"forma": t["forma"], "datos": b} for t, b in zip(s["tensores"], s["base64"], strict=True)]
    )
    assert all(np.array_equal(a, b) for a, b in zip(vuelta, pesos, strict=True))


@pytest.mark.parametrize("caso", _fixture()["rotacion"], ids=lambda c: f"semilla={c['semilla']}")
def test_rotacion(caso):
    n = pr.potencia_de_2(caso["largo"])
    # Los signos, idénticos: con uno distinto el servidor deshace otra rotación.
    assert pr.signos(caso["semilla"], n).astype(int).tolist() == caso["signos"]
    x = _f32(caso["x"])
    rotado = pr.rotar(x, caso["semilla"])
    assert len(rotado) == n
    np.testing.assert_allclose(rotado, _f32(caso["rotado"]), atol=caso["tolerancia"], rtol=0)
    np.testing.assert_allclose(pr.desrotar(rotado, caso["semilla"], caso["largo"]), x, atol=caso["tolerancia"], rtol=0)


def test_el_fixture_cubre_los_bordes():
    # Si se regenera con menos casos, que se note: son los bordes donde dos
    # implementaciones suelen separarse (escala cero, un solo valor, recorte en ±127,
    # cero negativo, relleno hasta la potencia de 2).
    nombres = {c["nombre"] for c in _fixture()["cuantizacion"]}
    assert {"todo_cero", "un_valor", "extremos", "matriz"} <= nombres
    largos = [c["largo"] for c in _fixture()["rotacion"]]
    assert any(n & (n - 1) for n in largos), "falta un largo que no sea potencia de 2"
