"""Particiones del simulador federado (federated/data.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "federated"))
import data


def _personas():
    rng = np.random.default_rng(0)
    ids = np.array(["P1"] * 60 + ["P2"] * 25 + ["P3"] * 16)
    y = rng.integers(0, 4, len(ids))
    return ids, y


def test_con_un_subcliente_es_la_particion_por_persona():
    ids, y = _personas()
    partes, nombres = data.partition_subclientes(ids, y, 1, seed=3)
    assert nombres == ["P1", "P2", "P3"]
    for a, b in zip(partes, data.partition_by_client(ids), strict=True):
        np.testing.assert_array_equal(a, b)


def test_no_mezcla_personas_y_cubre_todo_una_vez():
    ids, y = _personas()
    partes, nombres = data.partition_subclientes(ids, y, 4, seed=1)
    assert len(partes) == 12 and nombres[:4] == ["P1#0", "P1#1", "P1#2", "P1#3"]
    todos = np.concatenate(partes)
    assert sorted(todos.tolist()) == list(range(len(ids)))
    for p, n in zip(partes, nombres, strict=True):
        assert set(ids[p].tolist()) == {n.split("#")[0]}


def test_estratifica_por_clase_dentro_de_la_persona():
    ids, y = _personas()
    partes, nombres = data.partition_subclientes(ids, y, 4, seed=2)
    for persona in ("P1", "P2", "P3"):
        mias = [p for p, n in zip(partes, nombres, strict=True) if n.startswith(persona + "#")]
        for c in np.unique(y[ids == persona]):
            conteos = [int(np.sum(y[p] == c)) for p in mias]
            assert max(conteos) - min(conteos) <= 1, (persona, c, conteos)


def test_es_reproducible_y_depende_de_la_semilla():
    ids, y = _personas()
    a, _ = data.partition_subclientes(ids, y, 3, seed=5)
    b, _ = data.partition_subclientes(ids, y, 3, seed=5)
    c, _ = data.partition_subclientes(ids, y, 3, seed=6)
    assert all(np.array_equal(x, z) for x, z in zip(a, b, strict=True))
    assert any(not np.array_equal(x, z) for x, z in zip(a, c, strict=True))


def test_entradas_invalidas():
    ids, y = _personas()
    for s in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="sub_clientes"):
            data.partition_subclientes(ids, y, s)
    with pytest.raises(ValueError, match="largos"):
        data.partition_subclientes(ids, y[:-1], 2)
    with pytest.raises(ValueError, match="P3"):
        data.partition_subclientes(np.array(["P1"] * 10 + ["P3"] * 2), np.zeros(12, int), 3)
