"""La métrica σ de la propuesta, con casos calculados a mano."""
import numpy as np

from lsa import consistencia as cs


def test_distancias_euclideas():
    X = np.array([[0.0, 0.0], [3.0, 4.0], [0.0, 1.0]])
    D = cs.distancias(X)
    assert np.allclose(D, [[0, 5, 1], [5, 0, np.sqrt(9 + 9)], [1, np.sqrt(18), 0]])


def test_sigma_a_mano():
    # Muestras 0 y 1 cerca (distancia 1); la 2, lejos de todas.
    X = np.array([[0.0], [1.0], [10.0]])
    P = np.array([[0.9, 0.1], [0.5, 0.5], [0.2, 0.8]])
    r = cs.sigma(P, cs.distancias(X), epsilon=2.0)
    # N_0 = N_1 = {0, 1}; std sesgada por clase = |0.9-0.5|/2 = 0.2 en las dos clases.
    assert r["n_I"] == 2 and not r["en_I"][2]
    assert np.isclose(r["sigmas_i"][0], 0.2) and np.isclose(r["sigma"], 0.2)
    assert r["vecinos_medio"] == 2.0


def test_salidas_iguales_dan_cero_y_epsilon_chico_no_da_vecinos():
    X = np.array([[0.0], [0.5], [1.0]])
    P = np.tile([0.7, 0.2, 0.1], (3, 1))
    assert abs(cs.sigma(P, cs.distancias(X), 5.0)["sigma"]) < 1e-12
    r = cs.sigma(P, cs.distancias(X), 0.1)
    assert r["sigma"] is None and r["n_I"] == 0


def test_la_vecindad_incluye_a_la_muestra():
    # Con tres vecinos mutuos, m = 3 (la muestra cuenta), como en la definición.
    X = np.array([[0.0], [0.1], [0.2]])
    P = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    r = cs.sigma(P, cs.distancias(X), 1.0)
    esperado = np.std([1.0, 0.0, 1.0])  # 1/m con m = 3
    assert np.isclose(r["sigma"], esperado)


def test_epsilon_por_percentil():
    D = cs.distancias(np.arange(5.0)[:, None])  # distancias 1,1,1,1,2,2,2,3,3,4
    assert cs.epsilon_por_percentil(D, 50) == 2.0


def test_pureza_de_las_vecindades():
    D = cs.distancias(np.array([[0.0], [0.1], [5.0], [5.1]]))
    assert cs.pureza(D, np.array([0, 0, 1, 1]), 1.0) == 1.0
    assert cs.pureza(D, np.array([0, 1, 1, 1]), 1.0) == 0.5
    assert cs.pureza(D, np.array([0, 0, 1, 1]), 0.01) is None
