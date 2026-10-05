"""El seudónimo de participante tiene que salir igual en la app y en Python, o un
export viejo (con client_id crudo) y uno nuevo de la misma persona quedarían como
dos personas distintas en el experimento federado."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "federated"))
from data import SIN_CLIENTE, cargar_lsa_json, id_participante, partition_by_client, seudonimo_participante


def test_mismo_valor_que_kotlin():
    # El mismo valor fijo está en ParticipanteTest.kt de TraductorLSA.
    assert seudonimo_participante("user_2abc") == "p1_34a397c8c04e7f25"


def test_prioridades_de_id_participante():
    assert id_participante({"participant_id": "p1_aaaaaaaaaaaaaaaa", "client_id": "user_x"}) == "p1_aaaaaaaaaaaaaaaa"
    assert id_participante({"client_id": "user_2abc"}) == "p1_34a397c8c04e7f25"
    assert id_participante({"client_id": "p1_bbbbbbbbbbbbbbbb"}) == "p1_bbbbbbbbbbbbbbbb"
    assert id_participante({}) == SIN_CLIENTE


def test_export_viejo_y_nuevo_de_la_misma_persona_se_agrupan(tmp_path):
    seq = np.zeros((15, 126)).tolist()
    datos = {"t": 15, "d": 126, "by_date": {
        "2025-10-10": [{"label": "hola", "client_id": "user_2abc", "seq": seq}],
        "2026-10-05": [{"label": "hola", "participant_id": "p1_34a397c8c04e7f25", "seq": seq},
                       {"label": "bien", "participant_id": "p1_cccccccccccccccc", "seq": seq}],
    }}
    ruta = tmp_path / "lsa_samples.json"
    ruta.write_text(json.dumps(datos), encoding="utf-8")
    _, _, _, ids = cargar_lsa_json(ruta, verbose=False)
    assert ids.tolist()[0] == ids.tolist()[1] == "p1_34a397c8c04e7f25"
    assert len(partition_by_client(ids)) == 2
