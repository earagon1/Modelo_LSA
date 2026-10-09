"""El lector del dataset v2 entiende exactamente la línea que escribe la app
(`MuestraV2Test.kt` fija el mismo formato del lado de Kotlin)."""
import json

import numpy as np
import pytest

from lsa import contrato, dataset_v2


def _linea(lm_valor=0.5, n_frames=12, etiqueta="hola"):
    frame = {"t_ms": 0, "manos": [{"e": "Right", "s": 0.98, "lm": [lm_valor] * 63}]}
    return {"esquema": "senar-muestra-v2", "uuid": "u", "ts": 1, "label": etiqueta, "participant_id": "p1_x",
            "sesion": {"codigo": "P1", "orden": 1, "bloque": "A", "tanda": 1, "condicion": "luz=interior"},
            "dispositivo": "LENOVO TB321FU", "android": "16", "app": "1.0",
            "extractor": "mediapipe-tasks-vision-0.10.29", "camara": "frontal", "img_w": 480, "img_h": 640,
            "rotacion_deg": 270, "orientacion": "vertical", "fps": 13.0, "duracion_ms": 1100,
            "n_frames": n_frames, "frames": [frame] * n_frames}


def test_convierte_con_el_contrato_v2(tmp_path):
    ruta = tmp_path / "v2.jsonl"
    ruta.write_text(json.dumps(_linea()) + "\n\n", encoding="utf-8")
    muestras = list(dataset_v2.leer(ruta))
    assert len(muestras) == 1
    x = dataset_v2.secuencia(muestras[0])
    assert x.shape == (15, 126)
    # Right -> rh, y en vertical x queda comprimida al lienzo cuadrado: 0.5 -> 0.5.
    assert np.allclose(x[:, 63::3], 0.5) and np.allclose(x[:, :63], 0)
    assert np.allclose(x[:, 64::3], contrato.corregir_aspecto(np.full((21, 3), 0.5), 480, 640)[0, 1])


def test_linea_rota_corta_con_numero_de_linea(tmp_path):
    ruta = tmp_path / "v2.jsonl"
    ruta.write_text(json.dumps(_linea()) + "\n{roto\n", encoding="utf-8")
    with pytest.raises(ValueError, match=":2:"):
        list(dataset_v2.leer(ruta))
