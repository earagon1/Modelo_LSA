"""La guarda de los scripts heredados que pisan datos (auditoría, ML-M11)."""
import os
import subprocess
import sys

import pytest

from lsa import guarda

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_sin_nada_en_riesgo_no_pide_la_bandera():
    guarda.exigir_permiso("x.py", [], "otra cosa", argv=[])


def test_con_algo_en_riesgo_corta_y_dice_que_y_que_usar():
    with pytest.raises(SystemExit) as corte:
        guarda.exigir_permiso("x.py", ["models/words.json"], "scripts/entrenar_final.py", argv=[])
    texto = str(corte.value)
    assert "x.py" in texto
    assert "models/words.json" in texto
    assert "scripts/entrenar_final.py" in texto
    assert guarda.BANDERA in texto


def test_con_la_bandera_sigue():
    guarda.exigir_permiso("x.py", ["models/words.json"], "otra cosa", argv=[guarda.BANDERA])


def test_el_mensaje_no_lista_cientos_de_archivos():
    texto = guarda.mensaje("x.py", [f"a{i}.h5" for i in range(30)], "otra cosa", maximo=8)
    assert "a7.h5" in texto
    assert "a8.h5" not in texto
    assert "y 22 más" in texto


def test_existentes_deja_solo_lo_que_hay(tmp_path):
    hay = tmp_path / "hola.h5"
    hay.write_bytes(b"x")
    falta = tmp_path / "chau.h5"
    assert guarda.existentes([str(falta), str(hay)]) == [str(hay)]


def test_clases_sin_tomas_respeta_el_orden_de_words_json():
    clases = ["hola", "bien", "v", "l"]
    assert guarda.clases_sin_tomas(clases, ["l", "hola", "hola"]) == ["bien", "v"]
    assert guarda.clases_sin_tomas(clases, ["l", "hola", "v", "bien"]) == []


# Los scripts se corren de verdad, en una carpeta de prueba que imita el repo
# (constants.py arma las rutas desde el directorio actual). Solo estos dos se pueden
# correr acá: sus imports son livianos y la guarda está antes de que hagan nada.
# training_model.py y export_tflite.py importan TensorFlow y cargan el modelo.
def _correr(script, carpeta, *args):
    entorno = dict(os.environ, PYTHONPATH=RAIZ, PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, os.path.join(RAIZ, script), *args], cwd=carpeta, env=entorno,
                          capture_output=True, text=True, encoding="utf-8", timeout=300)


def _toma_de_20_frames(carpeta):
    toma = carpeta / "frame_actions" / "hola" / "sample_1"
    toma.mkdir(parents=True)
    for i in range(20):
        (toma / f"{i}.jpg").write_bytes(b"x")
    return toma


def test_normalize_samples_sin_la_bandera_no_borra_ningun_frame(tmp_path):
    pytest.importorskip("cv2")
    toma = _toma_de_20_frames(tmp_path)
    r = _correr("normalize_samples.py", tmp_path)
    assert r.returncode != 0
    assert "normalize_samples.py no corrió" in r.stderr
    assert sorted(os.listdir(toma)) == sorted(f"{i}.jpg" for i in range(20))


def test_normalize_samples_con_la_bandera_hace_lo_de_siempre(tmp_path):
    pytest.importorskip("cv2")
    toma = _toma_de_20_frames(tmp_path)
    r = _correr("normalize_samples.py", tmp_path, guarda.BANDERA)
    assert r.returncode == 0, r.stderr
    assert sorted(os.listdir(toma)) == [f"frame_{i:02}.jpg" for i in range(1, 16)]


def test_create_keypoints_sin_la_bandera_no_toca_el_h5(tmp_path):
    pytest.importorskip("mediapipe")
    pytest.importorskip("pandas")
    _toma_de_20_frames(tmp_path)
    h5 = tmp_path / "data" / "keypoints" / "hola.h5"
    h5.parent.mkdir(parents=True)
    h5.write_bytes(b"contenido original")
    r = _correr("create_keypoints.py", tmp_path)
    assert r.returncode != 0
    assert "hola.h5" in r.stderr
    assert h5.read_bytes() == b"contenido original"
