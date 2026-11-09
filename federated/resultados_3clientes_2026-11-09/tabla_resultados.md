Media +/- desvio sobre 5 replicas con distinta semilla.

| Escenario | Accuracy | F1-macro | F1-micro | sigma | Rondas | MB transferidos |
|---|---|---|---|---|---|---|
| Centralizado (techo) | 0.9367 +/- 0.0274 | 0.9370 +/- 0.0268 | 0.9367 +/- 0.0274 | 0.1152 +/- 0.0034 | - | - |
| Solo local (piso) | 0.8650 +/- 0.0191 | 0.8617 +/- 0.0230 | 0.8650 +/- 0.0191 | 0.1062 +/- 0.0036 | - | - |
| Solo local non-IID (piso) | 0.6719 +/- 0.0226 | 0.6033 +/- 0.0326 | 0.6719 +/- 0.0226 | 0.0911 +/- 0.0042 | - | - |
| FedAvg IID (3 clientes) | 0.9325 +/- 0.0140 | 0.9320 +/- 0.0145 | 0.9325 +/- 0.0140 | 0.1109 +/- 0.0039 | 30 | 115.9 |
| FedAvg non-IID (alpha=0.5) | 0.8958 +/- 0.0188 | 0.8946 +/- 0.0195 | 0.8958 +/- 0.0188 | 0.1033 +/- 0.0040 | 30 | 115.9 |

```

LECTURA DE LOS RESULTADOS
========================================================================
t-test pareado sobre 5 replicas, alpha=0.05.

Contra el PISO (cliente aislado con el MISMO tipo de reparto) -- es la
comparacion que justifica el enfoque federado:
  FedAvg IID (3 clientes) vs Solo local (piso): SUPERA al cliente aislado (+0.0675 +/- 0.0064, p=0.0004)
  FedAvg non-IID (alpha=0.5) vs Solo local non-IID (piso): SUPERA al cliente aislado (+0.2239 +/- 0.0117, p=0.0000)

Contra el TECHO (centralizado) -- cuanto cuesta no centralizar:
  FedAvg IID (3 clientes): alcanza al centralizado (diferencia no significativa) (-0.0042 +/- 0.0080, p=0.6306)
  FedAvg non-IID (alpha=0.5): queda 4.1 puntos por debajo del centralizado (-0.0408 +/- 0.0130, p=0.0348)

IID contra non-IID -- cuanto cuesta la heterogeneidad:
  el reparto non-IID cuesta 3.7 puntos (+0.0367 +/- 0.0090, p=0.0150)
```
