# Resultados para el informe

Generado por `scripts/reporte_tesis.py` desde `resultados_validacion_v2\2026-10-14_p1p2p3`. No editar a mano.

Datos: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}. Semillas: [0].

## 1. Escenarios de evaluación

| Escenario | Qué mide | Accuracy (%) | F1 macro (%) |
|---|---|---|---|
| Modelo original de la app | solo webcam, contrato v1; sobre las tomas reales de los teléfonos | 28,5 | 25,6 |
| Solo webcam, contrato v2 | mismo dato de entrenamiento, preprocesamiento corregido | 52,4 | 48,6 |
| Persona no vista (LOPO) | entrena sin la persona evaluada: la cifra honesta | 64,2 | 64,8 |
| Sesión no vista | agrupado por sesión de grabación | 77,3 | 75,9 |
| Esquema anterior | tomas mezcladas al azar: mide la fuga, no la generalización | 94,1 | 92,3 |

El modelo original muestra el 83,0 % de sus predicciones (confianza ≥ 0.7) y de esas acierta el 31,8 %: se equivoca con seguridad.

## 2. Fuga del esquema anterior

El esquema anterior da 94,1 %; con una persona no vista, 64,2 %. La diferencia, **30,0 puntos**, es lo que inflaba la validación original: tomas casi idénticas de la misma persona a ambos lados de la partición.

## 3. Por persona: modelo original contra persona no vista

| Persona | Modelo original (%) | Persona no vista (%) |
|---|---|---|
| P1 | 39,6 | 58,5 |
| P2 | 21,9 | 66,0 |
| P3 | 24,0 | 67,9 |

## 4. F1 por seña

| Seña | Modelo original | Persona no vista |
|---|---|---|
| hola | 40,0 | 86,3 |
| adios | 37,5 | 78,3 |
| bien | 22,7 | 40,0 |
| como estas | 48,5 | 62,2 |
| gracias | 13,8 | 62,9 |
| e | 25,8 | 51,0 |
| v | 0,0 | 79,1 |
| l | 0,0 | 56,5 |
| i | 0,0 | 65,2 |
| n | 7,7 | 94,1 |
| mi nombre es | 29,1 | 37,5 |
| por favor | 82,1 | 83,0 |
| nada | — | 46,7 |

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

## Tabla de escenarios en LaTeX (booktabs)

```latex
\begin{tabular}{llrr}
\toprule
Escenario & Qué mide & Accuracy (\%) & F1 macro (\%) \\
\midrule
Modelo original de la app & solo webcam, contrato v1; sobre las tomas reales de los teléfonos & 28,5 & 25,6 \\
Solo webcam, contrato v2 & mismo dato de entrenamiento, preprocesamiento corregido & 52,4 & 48,6 \\
Persona no vista (LOPO) & entrena sin la persona evaluada: la cifra honesta & 64,2 & 64,8 \\
Sesión no vista & agrupado por sesión de grabación & 77,3 & 75,9 \\
Esquema anterior & tomas mezcladas al azar: mide la fuga, no la generalización & 94,1 & 92,3 \\
\bottomrule
\end{tabular}
```
