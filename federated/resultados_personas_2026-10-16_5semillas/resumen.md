# Aprendizaje federado con personas reales

Rondas 40, épocas locales 2, semillas [0, 1, 2, 3, 4]. Se entrena y se sube solo la cabeza; el backbone sale de la webcam de P1 y está congelado. Agregación: fedavg; 1 cliente(s) por persona.

## A. Federado con una persona afuera

| | Accuracy (%) |
|---|---|
| Modelo del servidor, sin federar | 46,8 ± 11,3 |
| Piso: una sola persona | 62,3 ± 3,7 |
| FedAvg (2 personas) | 67,0 ± 4,4 |
| Lo mismo, con subida a 8 bits | 67,2 ± 4,5 |
| Techo: centralizado | 67,4 ± 5,0 |

Subida total de los clientes en toda la corrida: 4.24 MB en float32, 1.06 MB a 8 bits. Rondas hasta el 95 % del techo: 9.7 (n=15).

## B. Personalización de un usuario nuevo

Modelo global 65,2 % → personalizado con su tanda 1 69,3 % (mejora media 4,1 puntos; t pareada sobre la media de cada persona, n = 3: t = 2.43, p = 0.1355). Por persona: P1 7,2, P2 1,4, P3 3,8.

## C. Personalización de un miembro

Modelo global 65,7 % → personalizado con su tanda 1 66,7 % (mejora media 0,9 puntos; t pareada sobre la media de cada persona, n = 3: t = 0.62, p = 0.6010). Por persona: P1 3,4, P2 -1,7, P3 1,0.
