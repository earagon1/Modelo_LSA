Media +/- desvio sobre 5 replicas. Solo se comprime la subida;
la bajada del modelo global va siempre en float32.

| Codificacion del update | MB subida | Ahorro | MB total | Accuracy | F1-macro | Error del delta |
|---|---|---|---|---|---|---|
| sin comprimir (float32) | 96.5 | - | 193.1 | 0.8978 +/- 0.0193 | 0.8982 +/- 0.0194 | - |
| 8 bits estocastico | 24.1 | 75 % | 120.7 | 0.9120 +/- 0.0302 | 0.9119 +/- 0.0346 | 0.0137 |
| 4 bits estocastico | 12.1 | 87 % | 108.6 | 0.9043 +/- 0.0160 | 0.9061 +/- 0.0159 | 0.2489 |
| 2 bits estocastico | 6.0 | 94 % | 102.6 | 0.8967 +/- 0.0248 | 0.8982 +/- 0.0268 | 1.5137 |
| 4 bits determinista | 12.1 | 87 % | 108.6 | 0.8902 +/- 0.0288 | 0.8888 +/- 0.0339 | 0.1756 |

```

LECTURA DE LOS RESULTADOS
========================================================================
t-test pareado contra 'sin comprimir (float32)', 5 replicas, alpha=0.05.

  8 bits estocastico         ahorra 75 % de subida, sin perdida medible (p=0.3396)
  4 bits estocastico         ahorra 87 % de subida, sin perdida medible (p=0.5529)
  2 bits estocastico         ahorra 94 % de subida, sin perdida medible (p=0.9281)
  4 bits determinista        ahorra 87 % de subida, sin perdida medible (p=0.4391)

Estocastico contra determinista, a igual cantidad de bits:
  no hay diferencia significativa (+1.4 puntos, p=0.2714). Con esta cantidad de clientes y rondas el sesgo no alcanza a acumularse.
```
