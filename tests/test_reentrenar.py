"""Controles de reentrenar.py antes de publicar un paquete (auditoría, ML-M8)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import reentrenar as re_


def test_forzar_no_saltea_los_controles_tecnicos():
    for tecnico in ("sin_flex", "paridad", "senas_sin_test"):
        kw = {"sin_flex": True, "paridad": True, "senas_sin_test": []}
        kw[tecnico] = ["hola"] if tecnico == "senas_sin_test" else False
        tec, cal = re_.evaluar_controles(**kw, acc_nueva_comun=0.9, acc_referencia=0.8, tolerancia=0.01)
        assert not re_.se_publica(tec, cal, forzar=True), tecnico


def test_forzar_si_saltea_el_de_calidad():
    tec, cal = re_.evaluar_controles(True, True, [], acc_nueva_comun=0.70, acc_referencia=0.85, tolerancia=0.01)
    assert cal == {"no_empeora": False}
    assert not re_.se_publica(tec, cal, forzar=False)
    assert re_.se_publica(tec, cal, forzar=True)


def test_sin_paquete_previo_no_hay_con_que_comparar():
    tec, cal = re_.evaluar_controles(True, True, [], acc_nueva_comun=None, acc_referencia=None, tolerancia=0.01)
    assert re_.se_publica(tec, cal, forzar=False)


def test_una_sena_nueva_necesita_tomas_suficientes():
    previas = ["hola", "nada"]
    nuevas, postergadas = re_.senas_nuevas_aceptadas(previas, {"hola": 3, "chau": 25, "sol": 1}, minimo=20)
    assert nuevas == ["chau"]
    assert postergadas == {"sol": 1}


def test_ampliar_test_fijo_solo_con_las_senas_que_no_tienen():
    uuids = np.array([f"u{i}" for i in range(30)])
    y = np.array([0] * 10 + [1] * 20)
    test = {"u0", "u1"}  # la clase 0 ya tiene test
    agregados = re_.ampliar_test_fijo(test, uuids, y, ["hola", "chau"])
    assert len(agregados) == 3  # 15 % de 20
    assert all(int(u[1:]) >= 10 for u in agregados)
    assert re_.ampliar_test_fijo(test | set(agregados), uuids, y, ["hola", "chau"]) == []


def test_una_sena_con_una_sola_toma_no_se_mete_en_el_test():
    assert re_.ampliar_test_fijo(set(), np.array(["a"]), np.array([0]), ["sol"]) == []


def _toma(label="hola", pid="p_x"):
    return {"esquema": "senar-muestra-v2", "uuid": "u1", "participant_id": pid, "label": label,
            "img_w": 480, "img_h": 640, "frames": [{"manos": [{"e": "Left", "s": 0.9, "lm": [0.5] * 63}]}]}


def test_las_tomas_aportadas_se_revalidan(tmp_path):
    ruta = tmp_path / "p_x.jsonl"
    malas = [_toma(), {**_toma(), "label": "Holá"}, {**_toma(), "frames": "x"}]
    ruta.write_text("\n".join(json.dumps(m) for m in malas) + "\nno es json\n\n", encoding="utf-8")
    etiquetas, problemas = re_.leer_aportadas([str(ruta)])
    assert etiquetas == ["hola"]
    assert len(problemas) == 3
    assert problemas[0].startswith("p_x.jsonl:2:") and problemas[2] == "p_x.jsonl:4: no es JSON"


def test_cabeza_de_las_rondas(tmp_path):
    (tmp_path / "cabezas").mkdir()
    capas = [{"W": [[1.0, 0.0]], "b": [0.0, 0.0]}]
    (tmp_path / "cabezas" / "v6.json").write_text(json.dumps({"clases": ["a", "b"], "capas": capas}))
    (tmp_path / "estado.json").write_text(json.dumps({"paquete": 1, "version": 6}))
    version, pesos = re_.cabeza_de_las_rondas(str(tmp_path), 1)
    assert version == 6 and pesos[0].shape == (1, 2)
    # Las rondas son de otro paquete: no sirven de referencia para este.
    assert re_.cabeza_de_las_rondas(str(tmp_path), 2) is None
    assert re_.cabeza_de_las_rondas(str(tmp_path / "no_existe"), 1) is None
