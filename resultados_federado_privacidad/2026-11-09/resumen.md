# Privacidad diferencial central: acierto contra ε

DP-FedAvg, 5 rondas, C = 0,7, δ = 1e-5, semillas [0, 1, 2]. Acierto sobre la sesión 2 (media ± desvío entre semillas). La fila «sin DP» es FedAvg ponderado, sin recorte ni ruido.

| z | ε | 3 clientes | 15 clientes | 30 clientes |
|---|---|---|---|---|
| sin DP | - | 79,1 ± 1,3 | 78,9 ± 0,7 | 77,1 ± 0,0 |
| 0,0 | ∞ | 78,6 ± 0,6 | 78,9 ± 0,7 | 77,1 ± 0,0 |
| 0,1 | 357,3 | 79,1 ± 1,9 | 78,9 ± 0,7 | 77,4 ± 0,4 |
| 0,3 | 63,5 | 80,2 ± 2,9 | 78,6 ± 0,6 | 77,9 ± 0,6 |
| 1,0 | 13,2 | 12,2 ± 3,9 | 77,6 ± 1,6 | 78,1 ± 1,8 |
| 2,0 | 6,0 | 5,3 ± 1,6 | 59,3 ± 7,3 | 76,8 ± 1,0 |
| 4,0 | 2,8 | 3,3 ± 2,0 | 8,7 ± 1,4 | 53,4 ± 7,6 |

Norma media del promedio recortado sin ruido (la señal): 0.223. Parámetros de la cabeza: d = 13261.

**Estimación, no medición:** clientes necesarios para que la norma del ruido (z·C·√d / k) sea el 10 % de esa señal:

| z | ε (5 rondas) | clientes |
|---|---|---|
| 0,1 | 357,3 | 362 |
| 0,3 | 63,5 | 1085 |
| 1,0 | 13,2 | 3614 |
| 2,0 | 6,0 | 7228 |
| 4,0 | 2,8 | 14455 |
