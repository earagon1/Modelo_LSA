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

## 8. Robustez por condición (sesión 2)

Fuente: `resultados_robustez\2026-10-17_p1p2p3_5semillas`.

Generado por `scripts/robustez.py`. No editar a mano. Semillas: [0, 1, 2, 3, 4].

Entrenamiento: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}.

Sesión 2: 131 muestras, 13 clases | persona: {'P1': 48, 'P2': 47, 'P3': 36} | fuente: {'telefono': 131}.

Cada condición se mide con el mismo modelo que su referencia estándar. El IC 95 % es de Wilson sobre las tomas (cada toma promedia sus semillas); la caída tiene IC por bootstrap. Con 12 tomas por persona y condición los intervalos son anchos: las diferencias chicas no son concluyentes.

Correcciones aplicadas (de `correcciones_sesion2.json`): P3 órdenes 107–118 → lejos + poca luz. P3 no tiene «girado 30°» (descartada en la grabación) y a P2 le falta «por favor» a 30°.

## Reconocimiento

### Persona no vista (LOPO)

Entrena sin la persona evaluada. Estándar = sus tomas de la sesión 1.

| Condición | Tomas | Acierto (%) | IC 95 % | Caída vs estándar (pts, IC 95 %) | P1 | P2 | P3 | Confusión más común |
|---|---|---|---|---|---|---|---|---|
| estándar | 288 | 69,8 ± 4,1 | 64,3–74,8 | referencia | 60,6 | 73,1 | 75,6 | nada |
| poca luz | 36 | 60,0 ± 7,2 | 43,8–74,3 | -9,8 (-22,9 a +3,4) | 56,7 | 61,7 | 61,7 | nada |
| cerca | 36 | 62,2 ± 5,8 | 45,9–76,1 | -7,6 (-23,1 a +7,2) | 48,3 | 83,3 | 55,0 | v |
| lejos | 24 | 57,5 ± 9,0 | 38,1–74,9 | -12,3 (-27,6 a +3,1) | 63,3 | 51,7 | — | nada |
| lejos + poca luz | 12 | 61,7 ± 4,6 | 34,7–82,9 | -8,1 (-29,9 a +11,7) | — | — | 61,7 | por favor |
| girado 30° | 23 | 42,6 ± 3,6 | 24,9–62,4 | -27,2 (-44,5 a -9,1) | 61,7 | 21,8 | — | nada |

### Persona conocida, sesión nueva

Entrena con todo menos la tanda 2 de la persona. Estándar = esa tanda 2.

| Condición | Tomas | Acierto (%) | IC 95 % | Caída vs estándar (pts, IC 95 %) | P1 | P2 | P3 | Confusión más común |
|---|---|---|---|---|---|---|---|---|
| estándar | 144 | 84,6 ± 4,1 | 77,8–89,6 | referencia | 89,6 | 83,8 | 80,4 | bien |
| poca luz | 36 | 75,0 ± 5,2 | 58,9–86,2 | -9,6 (-21,7 a +1,9) | 85,0 | 70,0 | 70,0 | bien |
| cerca | 36 | 78,9 ± 3,7 | 63,1–89,1 | -5,7 (-18,2 a +6,0) | 86,7 | 90,0 | 60,0 | nada |
| lejos | 24 | 68,3 ± 4,8 | 48,3–83,3 | -16,3 (-32,1 a -0,8) | 75,0 | 61,7 | — | bien |
| lejos + poca luz | 12 | 73,3 ± 10,9 | 45,2–90,2 | -11,3 (-32,1 a +7,2) | — | — | 73,3 | por favor |
| girado 30° | 23 | 53,9 ± 6,6 | 34,5–72,2 | -30,7 (-48,0 a -12,9) | 76,7 | 29,1 | — | nada |

### La app en vivo

Modelo publicado (sha `276a79ea`), temperatura 1.20, umbral 0.75, sobre la secuencia que la app calculó en cada toma. Ya entrenó con la sesión 1 de estas personas, así que no hay referencia estándar justa.

| Condición | Tomas | Acierto (%) | Se muestra (%) | Acierto de lo mostrado (%) |
|---|---|---|---|---|
| poca luz | 36 | 83,3 | 75,0 | 100,0 |
| cerca | 36 | 83,3 | 69,4 | 92,0 |
| lejos | 24 | 79,2 | 45,8 | 100,0 |
| lejos + poca luz | 12 | 83,3 | 66,7 | 75,0 |
| girado 30° | 23 | 56,5 | 56,5 | 92,3 |

## Detección de manos (HandLandmarker)

| Condición | Tomas | Frames con mano (%) | Ambas manos en «por favor» y «cómo estás» (%) | FPS (mediana) |
|---|---|---|---|---|
| estándar | 288 | 100,0 | 58,4 | 8,1 |
| poca luz | 36 | 100,0 | 70,0 | 7,4 |
| cerca | 36 | 100,0 | 53,9 | 7,3 |
| lejos | 24 | 100,0 | 17,5 | 10,9 |
| lejos + poca luz | 12 | 100,0 | 55,0 | 6,3 |
| girado 30° | 23 | 100,0 | 21,1 | 13,2 |

Ojo al leer por condición: no todas tienen a las tres personas (ver las columnas por persona), así que una condición mezcla su efecto con el de quién y con qué dispositivo la grabó. Lo mismo pasa con los FPS: la mediana sube donde pesa más la tablet (16-18 FPS) y baja donde pesa el A55 (6 FPS).

Los frames de una toma guardada siempre tienen alguna mano (la captura se corta si se pierde): la degradación del extractor se ve en las dos manos y en los intentos cortados.

## Intentos cortados durante la grabación (aprox.)

| Persona | poca luz | cerca | lejos | girado 30° |
|---|---|---|---|---|
| P1 | 0 % | 0 % | 14 % | 29 % |
| P2 | 35 % | 15 % | 22 % | 69 % |
| P3 | 41 % | 8 % | — | 52 % |

Fuente: seguimiento en vivo y métricas de cada dispositivo (ver `correcciones_sesion2.json`).

## Tabla en LaTeX (booktabs)

```latex
\begin{tabular}{lrrrrr}
\toprule
Condición & \multicolumn{2}{c}{Persona no vista} & \multicolumn{2}{c}{Persona conocida} & Ambas manos (\%) \\
 & Acierto (\%) & Caída (pts) & Acierto (\%) & Caída (pts) & \\
\midrule
estándar & 69,8 & ref. & 84,6 & ref. & 58,4 \\
poca luz & 60,0 & -9,8 & 75,0 & -9,6 & 70,0 \\
cerca & 62,2 & -7,6 & 78,9 & -5,7 & 53,9 \\
lejos & 57,5 & -12,3 & 68,3 & -16,3 & 17,5 \\
lejos + poca luz & 61,7 & -8,1 & 73,3 & -11,3 & 55,0 \\
girado 30° & 42,6 & -27,2 & 53,9 & -30,7 & 21,1 \\
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
Solo webcam, contrato v2 & mismo dato de entrenamiento, preprocesamiento corregido & 56,0 $\pm$ 5,4 & 53,3 $\pm$ 5,3 \\
Persona no vista (LOPO) & entrena sin la persona evaluada: la cifra honesta & 67,3 $\pm$ 2,3 & 67,3 $\pm$ 2,3 \\
Sesión no vista & agrupado por sesión de grabación & 78,0 $\pm$ 7,1 & 75,1 $\pm$ 7,9 \\
Esquema anterior & tomas mezcladas al azar: mide la fuga, no la generalización & 94,3 $\pm$ 0,4 & 92,1 $\pm$ 0,5 \\
\bottomrule
\end{tabular}
```
