"""El reporte de la tesis: qué corrida de calibración va primero."""
import json
import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(RAIZ, "scripts"))

from reporte_tesis import corridas_de_calibracion, seccion_de_resumen  # noqa: E402


def _carpeta(tmp_path, corridas, origen=None):
    for c in corridas:
        (tmp_path / c).mkdir()
    if origen:
        (tmp_path / "calibracion_app.json").write_text(json.dumps({"origen": origen}), encoding="utf-8")
    return str(tmp_path)


def test_la_publicada_va_primero_aunque_no_sea_la_mas_reciente(tmp_path):
    carpeta = _carpeta(tmp_path, ["2026-10-16_lopo", "2026-11-10_lopo", "2026-11-14_temperatura"],
                       origen="resultados_calibracion/2026-10-16_lopo")
    orden = [(os.path.basename(c), pub) for c, pub in corridas_de_calibracion(carpeta)]
    assert orden == [("2026-10-16_lopo", True), ("2026-11-10_lopo", False), ("2026-11-14_temperatura", False)]


def test_publicada_del_medio_y_las_demas_por_fecha(tmp_path):
    carpeta = _carpeta(tmp_path, ["2026-10-16_lopo", "2026-11-10_lopo", "2026-11-14_temperatura"],
                       origen="resultados_calibracion/2026-11-10_lopo")
    assert [os.path.basename(c) for c, _ in corridas_de_calibracion(carpeta)] == [
        "2026-11-10_lopo", "2026-10-16_lopo", "2026-11-14_temperatura"]


def test_sin_calibracion_app_ninguna_es_la_publicada(tmp_path):
    carpeta = _carpeta(tmp_path, ["2026-10-16_lopo", "2026-11-10_lopo"])
    assert [pub for _, pub in corridas_de_calibracion(carpeta)] == [False, False]


def test_seccion_saca_el_titulo_propio_y_no_inventa_si_falta_el_resumen(tmp_path):
    assert seccion_de_resumen("7. X", str(tmp_path)) == []
    (tmp_path / "resumen.md").write_text("# Título propio\n\nCuerpo.\n", encoding="utf-8")
    s = seccion_de_resumen("7. X", str(tmp_path))
    assert s[1] == "## 7. X" and s[-1] == "Cuerpo." and "# Título propio" not in s
