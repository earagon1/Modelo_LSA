"""Tests del contrato de features v2: reglas sueltas + paridad contra el fixture
compartido con la app (contract/fixtures/paridad_features_v2.json)."""
import hashlib
import json
import os

import numpy as np
import pytest

from lsa import contrato
from lsa.contrato import Mano

FIXTURE = os.path.join(contrato.RUTA_CONTRATO, "fixtures", "paridad_features_v2.json")


def _mano(etq, score=0.9, valor=0.5):
    return Mano(etq, score, np.full((21, 3), valor, dtype=np.float32))


# --- reglas sueltas -------------------------------------------------------------

def test_aspecto_vertical_comprime_x_y_deja_y():
    lm = np.array([[0.0, 0.0, 0.1], [1.0, 1.0, -0.2]] + [[0.5, 0.5, 0.0]] * 19, dtype=np.float32)
    out = contrato.corregir_aspecto(lm, 480, 640)
    assert out[0, 0] == pytest.approx(0.125) and out[1, 0] == pytest.approx(0.875)
    assert out[0, 1] == pytest.approx(0.0) and out[1, 1] == pytest.approx(1.0)
    assert out[0, 2] == pytest.approx(0.1)  # z no se toca


def test_aspecto_horizontal_comprime_y():
    lm = np.zeros((21, 3), dtype=np.float32)
    out = contrato.corregir_aspecto(lm, 640, 480)
    assert out[0, 0] == pytest.approx(0.0) and out[0, 1] == pytest.approx(0.125)


def test_lateralidad_left_va_a_lh_right_a_rh():
    vec = contrato.frame_a_vector([_mano("Left", valor=0.3), _mano("Right", valor=0.7)], 640, 640)
    assert np.allclose(vec[:63:3], 0.3) and np.allclose(vec[63::3], 0.7)


def test_etiqueta_repetida_no_pisa_una_mano():
    vec = contrato.frame_a_vector([_mano("Right", 0.8, 0.3), _mano("Right", 0.95, 0.7)], 640, 640)
    assert np.allclose(vec[63::3], 0.7), "la de mayor score conserva rh"
    assert np.allclose(vec[:63:3], 0.3), "la otra pasa al slot libre en vez de perderse"


def test_frames_vacios_se_descartan():
    sec = np.zeros((6, 126), dtype=np.float32)
    sec[[1, 4], 0] = 1.0
    assert len(contrato.descartar_frames_vacios(sec)) == 2


@pytest.mark.parametrize("n", [16, 20, 27, 29, 30, 45, 100])
def test_submuestreo_entero_es_floor_exacto(n):
    sec = np.arange(n, dtype=np.float32)[:, None].repeat(126, 1)
    out = contrato.remuestrear_uniforme(sec)
    assert out[:, 0].tolist() == [float((i * n) // 15) for i in range(15)]


def test_interpolacion_respeta_extremos():
    sec = np.arange(10, dtype=np.float32)[:, None].repeat(126, 1)
    out = contrato.remuestrear_uniforme(sec)
    assert out[0, 0] == 0.0 and out[-1, 0] == 9.0
    assert np.all(np.diff(out[:, 0]) > 0)


def test_nunca_rellena_con_ceros():
    sec = np.ones((7, 126), dtype=np.float32)
    assert (contrato.remuestrear_uniforme(sec) != 0).all()


def test_secuencia_vacia_falla_en_vez_de_inventar():
    with pytest.raises(ValueError):
        contrato.remuestrear_uniforme(np.zeros((0, 126), dtype=np.float32))


def test_v1_con_pocos_frames_validos_se_excluye():
    sec = np.zeros((15, 126), dtype=np.float32)
    sec[:3, 0] = 1.0
    assert contrato.secuencia_desde_vectores_v1(sec) is None


# --- paridad contra el fixture compartido ----------------------------------------

def _cargar_fixture():
    with open(FIXTURE, encoding="utf-8") as f:
        return json.load(f)


def _frames(entrada):
    return [[Mano(m["etiqueta"], m["score"], np.asarray(m["landmarks"], dtype=np.float32))
             for m in fr["manos"]] for fr in entrada["frames"]]


@pytest.mark.parametrize("caso", _cargar_fixture()["casos"], ids=lambda c: c["nombre"])
def test_paridad_fixture(caso):
    fx = _cargar_fixture()
    e = caso["entrada"]
    frames = _frames(e)
    vectores = [contrato.frame_a_vector(f, e["ancho"], e["alto"]) for f in frames]
    np.testing.assert_allclose(vectores, caso["esperado"]["vectores_por_frame"], atol=fx["tolerancia"], rtol=0)
    out = contrato.secuencia_desde_frames(frames, e["ancho"], e["alto"])
    assert out.shape == (fx["T"], fx["D"]) and out.dtype == np.float32
    np.testing.assert_allclose(out, caso["esperado"]["secuencia"], atol=fx["tolerancia"], rtol=0)


def test_fixture_cubre_lo_que_pide_el_plan():
    casos = _cargar_fixture()["casos"]
    n_frames = {len(c["entrada"]["frames"]) for c in casos}
    assert {8, 10, 15, 20, 27} <= n_frames
    assert {c["entrada"]["orientacion"] for c in casos} >= {"vertical", "horizontal"}
    assert {c["entrada"]["camara"] for c in casos} >= {"frontal", "trasera"}


# --- espejo entre repos -----------------------------------------------------------

def test_contract_coincide_con_sha256sums():
    """Si este test falla, contract/ se editó sin regenerar SHA256SUMS, o la copia de
    este repo quedó distinta de la del otro. Regenerar con
    scripts/generar_fixtures_paridad.py y copiar contract/ a TraductorLSA."""
    with open(os.path.join(contrato.RUTA_CONTRATO, "SHA256SUMS"), encoding="utf-8") as f:
        esperados = dict(reversed(linea.strip().split("  ", 1)) for linea in f if linea.strip())
    for rel, sha in esperados.items():
        with open(os.path.join(contrato.RUTA_CONTRATO, rel), "rb") as fh:
            assert hashlib.sha256(fh.read()).hexdigest() == sha, rel
