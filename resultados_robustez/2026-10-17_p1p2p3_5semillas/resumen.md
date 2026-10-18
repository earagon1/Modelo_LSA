# Robustez por condición (sesión 2)

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
