"""Las piezas puras del análisis de robustez."""
import json

import numpy as np

from lsa import robustez as rb


def test_nombres_de_condicion():
    assert rb.nombre_condicion("luz=interior;angulo=0;distancia=media") == rb.ESTANDAR
    assert rb.nombre_condicion("luz=sombra;angulo=0;distancia=media") == "poca luz"
    assert rb.nombre_condicion("luz=interior;angulo=30;distancia=media") == "girado 30°"
    assert rb.nombre_condicion("luz=sombra;angulo=0;distancia=lejos") == "lejos + poca luz"
    assert all(c in rb.CONDICIONES for c in ("cerca", "lejos", "lejos + poca luz"))


def test_correcciones_cambian_solo_su_rango(tmp_path):
    ruta = tmp_path / "c.json"
    ruta.write_text(json.dumps({"descripcion": "x", "P3": {"condicion_real": [
        {"ordenes": [107, 118], "planilla": "luz=interior;angulo=0;distancia=lejos",
         "real": "luz=sombra;angulo=0;distancia=lejos"}]}}), encoding="utf-8")
    c = rb.leer_correcciones(str(ruta))
    lejos = "luz=interior;angulo=0;distancia=lejos"
    assert rb.condicion_efectiva("P3", 110, "B", lejos, c) == "lejos + poca luz"
    assert rb.condicion_efectiva("P3", 119, "B", lejos, c) == "lejos"
    assert rb.condicion_efectiva("P2", 110, "B", lejos, c) == "lejos"
    assert rb.condicion_efectiva("P3", 10, "A", lejos, c) == rb.ESTANDAR


def test_deteccion_cuenta_ambas_solo_en_senas_de_dos_manos():
    mano = {"e": "Right", "s": 0.9, "lm": [0.0] * 63}
    frames = [{"manos": [mano, mano]}, {"manos": [mano]}, {"manos": []}, {"manos": [mano]}]
    d = rb.deteccion({"label": "por favor", "frames": frames, "fps": 8.0})
    assert d["con_mano"] == 0.75 and d["ambas"] == 0.25 and d["frames"] == 4
    assert rb.deteccion({"label": "hola", "frames": frames})["ambas"] is None


def test_wilson_queda_en_cero_uno():
    lo, hi = rb.wilson(12, 12)
    assert 0.7 < lo < 1.0 and hi == 1.0
    lo, hi = rb.wilson(0, 12)
    assert lo == 0.0 and hi < 0.3


def test_acierto_por_muestra_promedia_semillas():
    preds = [{"id": "a", "y": 1, "pred": 1}, {"id": "a", "y": 1, "pred": 0}, {"id": "b", "y": 2, "pred": 2}]
    assert rb.acierto_por_muestra(preds) == {"a": 0.5, "b": 1.0}


def test_diferencia_bootstrap():
    a = np.ones(30)
    b = np.r_[np.ones(15), np.zeros(15)]
    d, lo, hi = rb.diferencia_bootstrap(a, b, repeticiones=2000)
    assert d == -0.5 and lo < -0.5 < hi < 0
