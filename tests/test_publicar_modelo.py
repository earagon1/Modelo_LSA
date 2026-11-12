"""publicar_modelo.py: no escribe nada si el modelo y sus clases no coinciden (auditoría, X3)."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import publicar_modelo as pm

CLASES = ["hola", "como estas", "nada"]


def _corrida(tmp_path, clases=CLASES, paridad_sha=None):
    desde = tmp_path / "run"
    desde.mkdir()
    (desde / pm.MODELO).write_bytes(b"modelo-falso")
    (desde / pm.PALABRAS).write_text(json.dumps({"word_ids": clases}), encoding="utf-8")
    if paridad_sha:
        (desde / pm.PARIDAD).write_text(json.dumps({"modelo_sha256": paridad_sha}), encoding="utf-8")
    cal = tmp_path / "calibracion_app.json"
    cal.write_text(json.dumps({"temperatura": 1.2, "umbral": 0.75, "origen": "x"}), encoding="utf-8")
    app = tmp_path / "app"
    for sub in ("main", "androidTest"):
        (app / "app" / "src" / sub / "assets").mkdir(parents=True)
    return desde, cal, app


def _forma(n):
    return lambda _modelo: ([1, pm.T, pm.D], [1, n], [])


def _assets(app):
    return app / "app" / "src" / "main" / "assets"


def test_publica_los_cuatro_y_lo_publicado_es_coherente(tmp_path):
    desde, cal, app = _corrida(tmp_path, paridad_sha=pm.sha256(b"modelo-falso"))
    man = pm.publicar(str(desde), str(cal), str(app), "prueba", "2026-11-11", forma_y_flex=_forma(3))
    assert man["clases"] == CLASES and man["n_salidas"] == 3 and man["features_version"] == pm.FEATURES_VERSION
    assert sorted(os.listdir(_assets(app))) == sorted([pm.MODELO, pm.PALABRAS, pm.CALIBRACION, pm.MANIFIESTO])
    assert json.loads((_assets(app) / pm.CALIBRACION).read_text())["modelo_sha256"] == pm.sha256(b"modelo-falso")
    assert (app / "app" / "src" / "androidTest" / "assets" / pm.PARIDAD).exists()
    assert pm.problemas_publicados(str(_assets(app))) == []


@pytest.mark.parametrize("forma, clases, esperado", [
    (([1, 15, 126], [1, 4], []), CLASES, "4 salidas"),
    (([1, 30, 126], [1, 3], []), CLASES, "entrada"),
    (([1, 15, 126], [1, 3], ["FlexTensorListReserve"]), CLASES, "Flex"),
    (([1, 15, 126], [1, 3], []), ["Hola", "como estás", "nada"], "nombre inválido"),
])
def test_si_algo_no_coincide_no_escribe_nada(tmp_path, forma, clases, esperado):
    desde, cal, app = _corrida(tmp_path, clases=clases)
    with pytest.raises(ValueError, match=esperado):
        pm.publicar(str(desde), str(cal), str(app), "prueba", "2026-11-11", forma_y_flex=lambda _m: forma)
    assert os.listdir(_assets(app)) == []


def test_calibracion_o_paridad_de_otro_modelo(tmp_path):
    desde, cal, app = _corrida(tmp_path, paridad_sha="0" * 64)
    with pytest.raises(ValueError, match="paridad"):
        pm.publicar(str(desde), str(cal), str(app), "p", "f", forma_y_flex=_forma(3))
    assert pm.problemas_de_calibracion({"temperatura": 1.2, "modelo_sha256": "otro"}, "este") == \
        ["la calibración dice que es de otro modelo (modelo_sha256 distinto)"]


def test_verificar_detecta_un_tflite_copiado_a_mano(tmp_path):
    desde, cal, app = _corrida(tmp_path)
    pm.publicar(str(desde), str(cal), str(app), "p", "f", forma_y_flex=_forma(3))
    (_assets(app) / pm.MODELO).write_bytes(b"otro-modelo")
    assert any(pm.MODELO in p for p in pm.problemas_publicados(str(_assets(app))))


def test_lo_publicado_hoy_en_la_app_es_coherente():
    assets = os.path.join(pm.APP_POR_DEFECTO, "app", "src", "main", "assets")
    if not os.path.exists(os.path.join(assets, pm.MANIFIESTO)):
        pytest.skip("sin la app al lado")
    assert pm.problemas_publicados(assets) == []
