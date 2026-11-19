# Simulación de las rondas del piloto federado

Generado por `federated/simular_piloto.py` desde `federated/piloto_sinP2`. No editar a mano.

4 rondas, dos clientes con las mismas tomas que el piloto (P2: 84, P1: 93), delta a 8 bits y FedAvg. Semillas [0, 1, 2, 3, 4]. Se evalúa a P2: sus 22 tomas habituales (del test fijo, nunca entrenan) y sus 47 tomas de la sesión 2. Una toma vale 4,5 y 2,1 puntos, respectivamente.

## Piloto real, versión por versión

| Versión | Habituales (%) | Sesión 2 (%) |
|---|---|---|
| v1 | 77,3 | 59,6 |
| v2 | 81,8 | 61,7 |
| v3 | 81,8 | 59,6 |
| v4 | 81,8 | 59,6 |
| v5 | 81,8 | 61,7 |

## Simulación, tras la última ronda

| Épocas locales | Habituales (%) | Sesión 2 (%) |
|---|---|---|
| 2 | 80,0 ± 2,5 (de 77,3 a 81,8) | 60,4 ± 1,2 (de 59,6 a 61,7) |
| 8 | 80,9 ± 2,0 (de 77,3 a 81,8) | 63,0 ± 1,9 (de 61,7 a 66,0) |
