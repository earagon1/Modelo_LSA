"""Medición de la duración de las tomas (scripts/duracion_captura.py, auditoría X4)."""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from duracion_captura import cotas_por_intervalo, estadisticas, instante_de_carpeta, medir_toma, resumir


def _toma(t_ms, app="1.0", fuente=None, manos=True, persona="P2", dispositivo="OnePlus CPH2663"):
    m = {"label": "hola", "dispositivo": dispositivo, "app": app, "fps": 7.5, "duracion_ms": t_ms[-1],
         "frames": [{"t_ms": t, "manos": [{"e": "Right"}] if manos else []} for t in t_ms],
         "sesion": {"codigo": persona, "bloque": "A"}}
    if fuente:
        m["fuente"] = fuente
    return m


def test_medir_toma_del_telefono():
    r = medir_toma(_toma([130, 260, 400, 530]))
    assert r["n_frames"] == 4 and r["n_con_mano"] == 4 and r["duracion_ms"] == 530
    assert r["entre_primero_y_ultimo_ms"] == 400 and r["primer_t_ms"] == 130
    assert r["marcas_reales"] and r["fuente"] == "telefono"


def test_la_webcam_migrada_no_tiene_marcas_reales():
    r = medir_toma(_toma([0, 33, 67], app="migrar_v1.py", fuente="webcam", manos=False))
    assert not r["marcas_reales"] and r["n_con_mano"] == 0


def test_estadisticas():
    e = estadisticas([1, 2, 3, 4, None])
    assert e["n"] == 4 and e["mediana"] == 2.5 and e["min"] == 1 and e["max"] == 4
    assert estadisticas([None]) is None


def test_resumir_agrupa_y_cuenta_frames():
    tomas = [medir_toma(_toma([100, 200])), medir_toma(_toma([100, 200, 300])),
             medir_toma(_toma([50, 100], persona="P1", dispositivo="LENOVO TB321FU"))]
    g = {x["persona"]: x for x in resumir(tomas, ["dispositivo", "persona"])}
    assert g["P2"]["tomas"] == 2 and g["P2"]["n_frames"] == {"2": 1, "3": 1}
    assert g["P2"]["duracion_ms"]["mediana"] == 250 and g["P1"]["tomas"] == 1


def test_instante_de_carpeta():
    assert instante_de_carpeta("sample_240917094224729441") == datetime(2024, 9, 17, 9, 42, 24, 729441)


def test_cotas_ordena_y_descarta_las_pausas():
    nombres = ["sample_240917094234000000", "sample_240917094224000000", "sample_240917094229500000",
               "sample_240917100000000000"]  # la última, 15 min después: otra tanda
    assert cotas_por_intervalo(nombres) == [5.5, 4.5]
