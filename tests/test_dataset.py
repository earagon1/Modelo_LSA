"""El cargador único arma persona, fuente y sesión como los necesita la validación."""
import itertools
import json

import numpy as np

from lsa import dataset as ds

_ids = itertools.count()


def _muestra(label, codigo, bloque="A", tanda=1, fuente=None, n_frames=12, manos=1, sesion_id=None):
    frame = {"t_ms": 0, "manos": [{"e": "Right", "s": 0.9, "lm": [0.5] * 63}] * manos}
    s = {"codigo": codigo, "orden": 1, "bloque": bloque, "tanda": tanda, "condicion": "x"}
    if sesion_id:
        s["sesion_id"] = sesion_id
    m = {"esquema": "senar-muestra-v2", "uuid": f"u{next(_ids)}", "ts": 1, "label": label, "participant_id": "p",
         "sesion": s, "dispositivo": "D", "android": "16", "app": "1", "extractor": "x", "camara": "frontal",
         "img_w": 480, "img_h": 640, "rotacion_deg": 0, "orientacion": "vertical", "fps": 10.0,
         "duracion_ms": 1000, "n_frames": n_frames, "frames": [frame] * n_frames}
    if fuente:
        m["fuente"] = fuente
    return m


def test_cargar_arma_los_metadatos(tmp_path, monkeypatch):
    web = tmp_path / "webcam.jsonl"
    web.write_text(json.dumps(_muestra("hola", "P1", "W", 0, "webcam", sesion_id="2025-10-23")) + "\n")
    tel = tmp_path / "captura" / "P2"
    tel.mkdir(parents=True)
    lineas = [_muestra("hola", "P2"), _muestra("nada", "P2", "N"), _muestra("bien", "PRUEBA"),
              _muestra("hola", "P2", n_frames=3), _muestra("hola", "P2", manos=0), _muestra("v", "P2", "B")]
    (tel / "lsa_samples_v2.jsonl").write_text("\n".join(json.dumps(m) for m in lineas) + "\n")
    monkeypatch.setattr(ds, "WEBCAM", str(web))
    monkeypatch.setattr(ds, "TELEFONO", str(tmp_path / "captura" / "*" / "lsa_samples_v2.jsonl"))

    d = ds.cargar()
    assert d.X.shape == (3, 15, 126)
    assert d.clases[-1] == "nada" and d.clases[d.y[2]] == "nada"
    assert d.persona.tolist() == ["P1", "P2", "P2"]
    assert d.fuente.tolist() == ["webcam", "telefono", "telefono"]
    assert d.sesion.tolist() == ["webcam-2025-10-23", "P2-A1", "P2-A2"]
    assert d.excluidas == {"prueba": 1, "pocos_frames": 2}
    sub = d.subconjunto(d.persona == "P2")
    assert len(sub.y) == 2 and np.all(sub.persona == "P2")


def test_robustez_solo_si_se_pide(tmp_path, monkeypatch):
    # La sesión 2 (bloque B) comparte archivo con la 1: no puede colarse en el entrenamiento.
    tel = tmp_path / "captura" / "P2"
    tel.mkdir(parents=True)
    lineas = [_muestra("hola", "P2"), _muestra("v", "P2", "B"), _muestra("l", "P2", "B")]
    (tel / "lsa_samples_v2.jsonl").write_text("\n".join(json.dumps(m) for m in lineas) + "\n")
    monkeypatch.setattr(ds, "WEBCAM", str(tmp_path / "no_existe.jsonl"))
    monkeypatch.setattr(ds, "TELEFONO", str(tmp_path / "captura" / "*" / "lsa_samples_v2.jsonl"))
    assert ds.cargar().bloque.tolist() == ["A"]
    b = ds.cargar(bloques=("B",))
    assert [b.clases[i] for i in b.y] == ["v", "l"]
    assert b.sesion.tolist() == ["P2-B", "P2-B"] and b.condicion.tolist() == ["x", "x"]


def test_sin_nada_respeta_words_json():
    con = ds.clases_del_modelo(incluir_nada=True)
    sin = ds.clases_del_modelo(incluir_nada=False)
    assert con[:-1] == sin and con[-1] == "nada"


def test_aportadas_sin_sesion_y_sin_repetir(tmp_path, monkeypatch):
    # Lo que manda la app al servidor: sin sesión guiada, con participant_id y uuid propios.
    tel = tmp_path / "captura" / "P2"
    tel.mkdir(parents=True)
    base = _muestra("hola", "P2")
    base["uuid"] = "u-1"
    (tel / "lsa_samples_v2.jsonl").write_text(json.dumps(base) + "\n")
    nueva = _muestra("chau", "x")
    nueva.pop("sesion")
    nueva.update(uuid="u-2", participant_id="p1_abc")
    repetida = dict(base)  # la misma toma, aportada otra vez
    aporte = tmp_path / "aporte.jsonl"
    aporte.write_text(json.dumps(nueva) + "\n" + json.dumps(repetida) + "\n")
    monkeypatch.setattr(ds, "WEBCAM", str(tmp_path / "no_existe.jsonl"))
    monkeypatch.setattr(ds, "TELEFONO", str(tmp_path / "captura" / "*" / "lsa_samples_v2.jsonl"))

    d = ds.cargar(aportadas=(str(aporte),), clases=["hola", "chau", "nada"])
    assert [d.clases[i] for i in d.y] == ["hola", "chau"]
    assert d.persona.tolist() == ["P2", "p1_abc"] and d.fuente.tolist() == ["telefono", "aporte"]
    assert d.sesion.tolist()[1] == "aporte-p1_abc" and d.uuid.tolist() == ["u-1", "u-2"]
    assert d.excluidas == {"repetidas": 1}
    # Sin la clase nueva, la toma aportada se descarta en vez de cruzar etiquetas.
    assert ds.cargar(aportadas=(str(aporte),), clases=["hola", "nada"]).excluidas["clase_desconocida"] == 1
