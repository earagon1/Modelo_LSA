"""Las piezas puras del análisis de robustez."""
import json

import numpy as np
import pytest

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


def _pred(persona, toma, condicion, semilla, acierta):
    return {"id": f"{persona}-{toma}", "persona": persona, "condicion": condicion, "semilla": semilla,
            "y": 1, "pred": 1 if acierta else 0}


def test_matrices_por_persona_son_tomas_por_semillas():
    preds = [_pred("P1", 1, "cerca", 0, True), _pred("P1", 1, "cerca", 1, False),
             _pred("P1", 2, "cerca", 0, True), _pred("P1", 2, "cerca", 1, True),
             _pred("P2", 1, "cerca", 0, False), _pred("P2", 1, "cerca", 1, False),
             _pred("P1", 3, "lejos", 0, True), _pred("P1", 3, "lejos", 1, True)]
    m = rb.matrices_por_persona(preds, "cerca")
    assert sorted(m) == ["P1", "P2"]
    assert m["P1"].tolist() == [[1, 0], [1, 1]] and m["P2"].tolist() == [[0, 0]]
    # A una toma le falta una semilla: es un error de la corrida y tiene que notarse.
    with pytest.raises(KeyError):
        rb.matrices_por_persona([*preds, _pred("P2", 2, "cerca", 0, True)], "cerca")


def test_acierto_con_ic_pesa_cada_toma_y_contiene_la_media():
    m = {"P1": np.ones((30, 3)), "P2": np.r_[np.ones((5, 3)), np.zeros((5, 3))]}
    media, lo, hi = rb.acierto_con_ic(m, repeticiones=2000)
    assert media == pytest.approx(35 / 40)
    assert lo < media < hi <= 1.0
    # Sin variación no hay intervalo: todas las tomas y semillas aciertan.
    assert rb.acierto_con_ic({"P1": np.ones((12, 5))}, repeticiones=200) == (1.0, 1.0, 1.0)


def test_el_ic_cuenta_la_variacion_entre_semillas():
    # Cada semilla acierta todo o nada: remuestrear solo las tomas daría un intervalo
    # de ancho cero alrededor de 0,6, como si el modelo fuera uno solo (ML-M3).
    m = {"P1": np.tile(np.array([1.0, 1.0, 1.0, 0.0, 0.0]), (12, 1))}
    media, lo, hi = rb.acierto_con_ic(m, repeticiones=2000)
    assert media == pytest.approx(0.6) and lo < 0.25 and hi == 1.0


def test_la_caida_se_mide_contra_el_estandar_de_las_mismas_personas():
    # P1 acierta 50 % en estándar y P2 100 %. La condición la grabó solo P1, sin caer:
    # contra el estándar de los dos daría -25 puntos que no son de la condición (ML-M6).
    ref = {"P1": np.r_[np.ones((10, 2)), np.zeros((10, 2))], "P2": np.ones((20, 2))}
    cond = {"P1": np.r_[np.ones((6, 2)), np.zeros((6, 2))]}
    c = rb.caida_pareada(ref, cond, repeticiones=2000)
    assert c["caida"] == pytest.approx(0.0) and c["referencia"] == pytest.approx(0.5)
    assert c["por_persona"] == {"P1": pytest.approx(0.0)}
    assert c["ic95"][0] < 0 < c["ic95"][1]


def test_la_caida_promedia_personas_no_tomas():
    ref = {"P1": np.ones((10, 1)), "P2": np.ones((10, 1))}
    cond = {"P1": np.ones((12, 1)), "P2": np.zeros((4, 1))}  # P2 cae entero, con menos tomas
    c = rb.caida_pareada(ref, cond, repeticiones=500)
    assert c["por_persona"] == {"P1": 0.0, "P2": -1.0} and c["caida"] == -0.5
    assert c["ic95"] == (-0.5, -0.5)
    # Una persona sin referencia no entra; sin ninguna, no hay caída que informar.
    assert rb.caida_pareada(ref, {"P9": np.ones((3, 1))}) is None
