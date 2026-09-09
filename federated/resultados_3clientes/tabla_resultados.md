Media +/- desvio sobre 5 replicas con distinta semilla.

| Escenario | Accuracy | F1-macro | F1-micro | sigma | Rondas | MB transferidos |
|---|---|---|---|---|---|---|
| Centralizado (techo) | 0.9641 +/- 0.0122 | 0.9650 +/- 0.0125 | 0.9641 +/- 0.0122 | 0.0984 +/- 0.0017 | - | - |
| Solo local (piso) | 0.8946 +/- 0.0272 | 0.8954 +/- 0.0283 | 0.8946 +/- 0.0272 | 0.0881 +/- 0.0049 | - | - |
| FedAvg IID (3 clientes) | 0.9391 +/- 0.0087 | 0.9418 +/- 0.0079 | 0.9391 +/- 0.0087 | 0.0922 +/- 0.0030 | 30 | 115.9 |
| FedAvg non-IID (alpha=0.5) | 0.9130 +/- 0.0283 | 0.9160 +/- 0.0274 | 0.9130 +/- 0.0283 | 0.0862 +/- 0.0043 | 30 | 115.9 |

```

LECTURA DE LOS RESULTADOS
========================================================================
t-test pareado sobre 5 replicas, alpha=0.05.

Contra el PISO (cliente aislado) -- es la comparacion que
justifica el enfoque federado:
  FedAvg IID (3 clientes): SUPERA al cliente aislado (+0.0446 +/- 0.0159, p=0.0485)
  FedAvg non-IID (alpha=0.5): no se distingue del cliente aislado (+0.0185 +/- 0.0259, p=0.5145)

Contra el TECHO (centralizado) -- cuanto cuesta no centralizar:
  FedAvg IID (3 clientes): queda 2.5 puntos por debajo del centralizado (-0.0250 +/- 0.0056, p=0.0111)
  FedAvg non-IID (alpha=0.5): queda 5.1 puntos por debajo del centralizado (-0.0511 +/- 0.0136, p=0.0198)

IID contra non-IID -- cuanto cuesta la heterogeneidad:
  la diferencia entre IID y non-IID no es significativa (+0.0261 +/- 0.0108, p=0.0729)
```
