# Robustez por condición (sesión 2)

Generado por `scripts/robustez.py`. No editar a mano. Semillas: [0, 1, 2, 3, 4].

Entrenamiento: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}.

Sesión 2: 131 muestras, 13 clases | persona: {'P1': 48, 'P2': 47, 'P3': 36} | fuente: {'telefono': 131}.

Cada condición se mide con el mismo modelo que su referencia estándar. La caída se calcula persona por persona, contra el estándar de esa misma persona, y se promedia entre las personas que tienen la condición (no todas las condiciones tienen a las tres). Los IC 95 % son por bootstrap de las tomas dentro de cada persona y de las semillas (10 000 remuestreos).

Los intervalos valen para estas personas: con tres o menos no se puede estimar cuánto cambia el resultado de una persona a otra, y por eso cada tabla muestra a cada persona por separado. Con 12 tomas por persona y condición los intervalos son anchos: las diferencias chicas no son concluyentes.

Recalculado desde las predicciones de `resultados_robustez/2026-10-17_p1p2p3_5semillas`, sin volver a entrenar.

Correcciones aplicadas (de `correcciones_sesion2.json`): P3 órdenes 107–118 → lejos + poca luz. P3 no tiene «girado 30°» (descartada en la grabación) y a P2 le falta «por favor» a 30°.

## Reconocimiento

### Persona no vista (LOPO)

Entrena sin la persona evaluada. Estándar = sus tomas de la sesión 1.

| Condición | Tomas | Acierto (%) | IC 95 % | Caída vs estándar de las mismas personas (pts, IC 95 %) | P1: acierto (caída) | P2: acierto (caída) | P3: acierto (caída) | Confusión más común |
|---|---|---|---|---|---|---|---|---|
| estándar | 288 | 69,8 ± 4,1 | 64,1–75,3 | referencia | 60,6 | 73,1 | 75,6 | nada |
| poca luz | 36 | 60,0 ± 7,2 | 45,6–73,9 | -9,8 (-24,2 a +4,3), ref. 69,8 | 56,7 (-4,0) | 61,7 (-11,5) | 61,7 (-14,0) | nada |
| cerca | 36 | 62,2 ± 5,8 | 47,8–76,1 | -7,6 (-22,5 a +6,7), ref. 69,8 | 48,3 (-12,3) | 83,3 (+10,2) | 55,0 (-20,6) | v |
| lejos | 24 | 57,5 ± 9,0 | 40,0–73,3 | -9,4 (-25,9 a +6,9), ref. 66,9 | 63,3 (+2,7) | 51,7 (-21,5) | — | nada |
| lejos + poca luz | 12 | 61,7 ± 4,6 | 38,3–83,3 | -14,0 (-38,3 a +9,6), ref. 75,6 | — | — | 61,7 (-14,0) | por favor |
| girado 30° | 23 | 42,6 ± 3,6 | 26,1–59,1 | -25,1 (-42,9 a -7,6), ref. 66,9 | 61,7 (+1,0) | 21,8 (-51,3) | — | nada |

### Persona conocida, sesión nueva

Entrena con todo menos la tanda 2 de la persona. Estándar = esa tanda 2.

| Condición | Tomas | Acierto (%) | IC 95 % | Caída vs estándar de las mismas personas (pts, IC 95 %) | P1: acierto (caída) | P2: acierto (caída) | P3: acierto (caída) | Confusión más común |
|---|---|---|---|---|---|---|---|---|
| estándar | 144 | 84,6 ± 4,1 | 78,3–90,6 | referencia | 89,6 | 83,8 | 80,4 | bien |
| poca luz | 36 | 75,0 ± 5,2 | 62,8–86,1 | -9,6 (-23,2 a +3,1), ref. 84,6 | 85,0 (-4,6) | 70,0 (-13,8) | 70,0 (-10,4) | bien |
| cerca | 36 | 78,9 ± 3,7 | 67,2–89,4 | -5,7 (-19,0 a +6,5), ref. 84,6 | 86,7 (-2,9) | 90,0 (+6,2) | 60,0 (-20,4) | nada |
| lejos | 24 | 68,3 ± 4,8 | 51,7–83,3 | -18,3 (-35,6 a -2,3), ref. 86,7 | 75,0 (-14,6) | 61,7 (-22,1) | — | bien |
| lejos + poca luz | 12 | 73,3 ± 10,9 | 50,0–93,3 | -7,1 (-34,6 a +17,5), ref. 80,4 | — | — | 73,3 (-7,1) | bien |
| girado 30° | 23 | 53,9 ± 6,6 | 38,3–69,6 | -33,8 (-50,5 a -17,4), ref. 86,7 | 76,7 (-12,9) | 29,1 (-54,7) | — | nada |

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
lejos & 57,5 & -9,4 & 68,3 & -18,3 & 17,5 \\
lejos + poca luz & 61,7 & -14,0 & 73,3 & -7,1 & 55,0 \\
girado 30° & 42,6 & -25,1 & 53,9 & -33,8 & 21,1 \\
\bottomrule
\end{tabular}
```
