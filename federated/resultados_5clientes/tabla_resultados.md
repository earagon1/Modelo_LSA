Media +/- desvio sobre 5 replicas con distinta semilla.

| Escenario | Accuracy | F1-macro | F1-micro | sigma | Rondas | MB transferidos |
|---|---|---|---|---|---|---|
| Centralizado (techo) | 0.9641 +/- 0.0122 | 0.9650 +/- 0.0125 | 0.9641 +/- 0.0122 | 0.0984 +/- 0.0017 | - | - |
| Solo local (piso) | 0.8359 +/- 0.0468 | 0.8321 +/- 0.0471 | 0.8359 +/- 0.0468 | 0.0829 +/- 0.0087 | - | - |
| FedAvg IID (5 clientes) | 0.8989 +/- 0.0222 | 0.8998 +/- 0.0253 | 0.8989 +/- 0.0222 | 0.0870 +/- 0.0029 | 30 | 193.1 |
| FedAvg non-IID (alpha=0.5) | 0.8522 +/- 0.0210 | 0.8444 +/- 0.0291 | 0.8522 +/- 0.0210 | 0.0784 +/- 0.0040 | 30 | 193.1 |

```

LECTURA DE LOS RESULTADOS
========================================================================
t-test pareado sobre 5 replicas, alpha=0.05.

Contra el PISO (cliente aislado) -- es la comparacion que
justifica el enfoque federado:
  FedAvg IID (5 clientes): SUPERA al cliente aislado (+0.0630 +/- 0.0201, p=0.0349)
  FedAvg non-IID (alpha=0.5): no se distingue del cliente aislado (+0.0163 +/- 0.0278, p=0.5886)

Contra el TECHO (centralizado) -- cuanto cuesta no centralizar:
  FedAvg IID (5 clientes): queda 6.5 puntos por debajo del centralizado (-0.0652 +/- 0.0077, p=0.0011)
  FedAvg non-IID (alpha=0.5): queda 11.2 puntos por debajo del centralizado (-0.1120 +/- 0.0136, p=0.0012)

IID contra non-IID -- cuanto cuesta la heterogeneidad:
  el reparto non-IID cuesta 4.7 puntos (+0.0467 +/- 0.0154, p=0.0388)
```
