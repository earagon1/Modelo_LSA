# Resultados para el informe

Generado por `scripts/reporte_tesis.py` desde `resultados_validacion_v2\2026-10-16_p1p2p3_5semillas`. No editar a mano.

Datos: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}. Semillas: [0, 1, 2, 3, 4].

## 1. Escenarios de evaluación

| Escenario | Qué mide | Accuracy (%) | F1 macro (%) |
|---|---|---|---|
| Modelo original de la app | solo webcam, contrato v1; sobre las tomas reales de los teléfonos | 28,5 | 25,6 |
| Solo webcam, contrato v2 | mismo dato de entrenamiento, preprocesamiento corregido | 56,0 ± 5,4 | 53,3 ± 5,3 |
| Persona no vista (LOPO) | entrena sin la persona evaluada: la cifra honesta | 67,3 ± 2,3 | 67,3 ± 2,3 |
| Sesión no vista | agrupado por sesión de grabación | 78,0 ± 7,1 | 75,1 ± 7,9 |
| Esquema anterior | tomas mezcladas al azar: mide la fuga, no la generalización | 94,3 ± 0,4 | 92,1 ± 0,5 |

El modelo original muestra el 83,0 % de sus predicciones (confianza ≥ 0.7) y de esas acierta el 31,8 %: se equivoca con seguridad.

## 2. Fuga del esquema anterior

El esquema anterior da 94,3 %; con una persona no vista, 67,3 %. La diferencia, **27,0 puntos**, es lo que inflaba la validación original: tomas casi idénticas de la misma persona a ambos lados de la partición.

## 3. Por persona: modelo original contra persona no vista

| Persona | Modelo original (%) | Persona no vista (%) |
|---|---|---|
| P1 | 39,6 | 60,0 |
| P2 | 21,9 | 73,0 |
| P3 | 24,0 | 68,9 |

## 4. F1 por seña

| Seña | Modelo original | Persona no vista |
|---|---|---|
| hola | 40,0 | 91,3 |
| adios | 37,5 | 74,7 |
| bien | 22,7 | 46,4 |
| como estas | 48,5 | 56,9 |
| gracias | 13,8 | 46,9 |
| e | 25,8 | 50,4 |
| v | 0,0 | 79,8 |
| l | 0,0 | 76,3 |
| i | 0,0 | 78,1 |
| n | 7,7 | 97,1 |
| mi nombre es | 29,1 | 48,7 |
| por favor | 82,1 | 76,8 |
| nada | — | 51,9 |

## 5. Sesión de captura por dispositivo

| | P1 | P2 | P3 |
|---|---|---|---|
| Dispositivo | LENOVO TB321FU | OnePlus CPH2663 | samsung SM-A556E |
| Android | 16 | 16 | 16 |
| Tomas | 106 | 106 | 106 |
| FPS (mediana) | 17,1 | 7,7 | 6,3 |
| FPS (mín–máx) | 11,3–18,0 | 6,1–14,0 | 4,8–12,1 |
| Frames por toma | 15: 106 | 10: 62, 15: 44 | 10: 96, 15: 10 |
| Duración de una captura (ms, mediana) | 876 | 1418 | 1612 |
| Duración de la sesión (min) | 9,8 | 19,4 | 30,0 |

Lectura: la app captura una cantidad fija de frames según los FPS (10, 15 o 20), así que
la ventana de tiempo que cubre una toma depende del dispositivo. En un equipo rápido 15 frames
pueden durar menos de un segundo, y una seña larga no entra completa; en uno lento la captura
baja a 10 frames. Es el argumento empírico para que el contrato v2 capture por tiempo.

## % de frames con las dos manos detectadas, por seña

| Seña | P1 | P2 | P3 |
|---|---|---|---|
| adios | 0 % | 0 % | 0 % |
| bien | 0 % | 0 % | 1 % |
| como estas | 52 % | 24 % | 54 % |
| e | 0 % | 0 % | 0 % |
| gracias | 0 % | 1 % | 1 % |
| hola | 0 % | 0 % | 5 % |
| i | 0 % | 0 % | 0 % |
| l | 0 % | 0 % | 0 % |
| mi nombre es | 0 % | 0 % | 12 % |
| n | 0 % | 0 % | 0 % |
| nada | 35 % | 21 % | 24 % |
| por favor | 78 % | 55 % | 88 % |
| v | 0 % | 0 % | 8 % |

## Versión LaTeX (booktabs)

```latex
\begin{tabular}{lrrr}
\toprule
 & P1 & P2 & P3 \\
\midrule
Dispositivo & LENOVO TB321FU & OnePlus CPH2663 & samsung SM-A556E \\
Android & 16 & 16 & 16 \\
Tomas & 106 & 106 & 106 \\
FPS (mediana) & 17,1 & 7,7 & 6,3 \\
FPS (mín–máx) & 11,3–18,0 & 6,1–14,0 & 4,8–12,1 \\
Frames por toma & 15: 106 & 10: 62, 15: 44 & 10: 96, 15: 10 \\
Duración de una captura (ms, mediana) & 876 & 1418 & 1612 \\
Duración de la sesión (min) & 9,8 & 19,4 & 30,0 \\
\bottomrule
\end{tabular}
```

## 6. Aprendizaje federado con personas reales

Fuente: `federated\resultados_personas_2026-10-16_5semillas`.

Rondas 40, épocas locales 2, semillas [0, 1, 2, 3, 4]. Se entrena y se sube solo la cabeza; el backbone sale de la webcam de P1 y está congelado.

## A. Federado con una persona afuera

| | Accuracy (%) |
|---|---|
| Modelo del servidor, sin federar | 46,8 ± 11,3 |
| Piso: una sola persona | 62,3 ± 3,7 |
| FedAvg (2 personas) | 67,0 ± 4,4 |
| FedAvg con subida a 8 bits | 67,2 ± 4,5 |
| Techo: centralizado | 67,4 ± 5,0 |

Subida total de los clientes en toda la corrida: 4.24 MB en float32, 1.06 MB a 8 bits. Rondas hasta el 95 % del techo: 9.7 (n=15).

## B. Personalización de un usuario nuevo

Modelo global 65,2 % → personalizado con su tanda 1 69,3 % (mejora media 4,1 puntos; t = 1.87, p = 0.0832).

## C. Personalización de un miembro

Modelo global 65,7 % → personalizado con su tanda 1 66,7 % (mejora media 0,9 puntos; t = 0.82, p = 0.4282).

## 7. Calibración y umbral de la app

Fuente: `resultados_calibracion\2026-10-16_lopo`.

Datos: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}. Semillas: [0, 1, 2].

- Temperatura para la app (mediana de 9 pliegues): **T = 1.20**
- ECE: 17,1 % sin calibrar → 12,7 % calibrado
- Con el umbral actual de la app (0,70, sin calibrar): se muestra el 69,1 % y acierta el 77,5 % de lo mostrado.
- Umbral elegido (acierto objetivo 85,0 %): **0.85** sobre la confianza calibrada: muestra el 49,7 % y acierta el 87,1 % de lo mostrado.

| Umbral | Mostradas sin calibrar | Acierto | Mostradas calibrado | Acierto |
|---|---|---|---|---|
| 0.30 | 91,5 | 67,4 | 91,2 | 67,5 |
| 0.35 | 90,5 | 67,9 | 89,7 | 68,2 |
| 0.40 | 89,5 | 68,4 | 88,6 | 68,5 |
| 0.45 | 88,3 | 68,8 | 86,3 | 69,6 |
| 0.50 | 86,1 | 70,2 | 81,7 | 72,5 |
| 0.55 | 81,7 | 72,7 | 76,5 | 74,0 |
| 0.60 | 77,0 | 74,0 | 71,8 | 76,4 |
| 0.65 | 72,7 | 75,9 | 66,7 | 78,6 |
| 0.70 | 69,1 | 77,5 | 62,9 | 80,8 |
| 0.75 | 65,3 | 80,1 | 59,6 | 83,0 |
| 0.80 | 62,4 | 81,3 | 55,7 | 84,4 |
| 0.85 | 57,9 | 83,2 | 49,7 | 87,1 |
| 0.90 | 52,1 | 85,7 | 44,1 | 88,8 |
| 0.95 | 43,3 | 87,9 | 32,9 | 89,2 |

## Tabla de escenarios en LaTeX (booktabs)

```latex
\begin{tabular}{llrr}
\toprule
Escenario & Qué mide & Accuracy (\%) & F1 macro (\%) \\
\midrule
Modelo original de la app & solo webcam, contrato v1; sobre las tomas reales de los teléfonos & 28,5 & 25,6 \\
Solo webcam, contrato v2 & mismo dato de entrenamiento, preprocesamiento corregido & 56,0 $\pm$ 5,4 & 53,3 $\pm$ 5,3 \\
Persona no vista (LOPO) & entrena sin la persona evaluada: la cifra honesta & 67,3 $\pm$ 2,3 & 67,3 $\pm$ 2,3 \\
Sesión no vista & agrupado por sesión de grabación & 78,0 $\pm$ 7,1 & 75,1 $\pm$ 7,9 \\
Esquema anterior & tomas mezcladas al azar: mide la fuga, no la generalización & 94,3 $\pm$ 0,4 & 92,1 $\pm$ 0,5 \\
\bottomrule
\end{tabular}
```
