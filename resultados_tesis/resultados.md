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

Fuente: `resultados_calibracion\2026-11-10_lopo`.

Datos: 1409 muestras, 13 clases | persona: {'P1': 1197, 'P2': 106, 'P3': 106} | fuente: {'telefono': 318, 'webcam': 1091} | excluidas: {'pocos_frames': 109}. Semillas: [0, 1, 2].

- Temperatura para la app (mediana de 9 pliegues): **T = 1.29**
- ECE: 18,6 % sin calibrar → 13,8 % calibrado
- Con el umbral actual de la app (0,70, sin calibrar): se muestra el 70,9 % y acierta el 77,1 % de lo mostrado.
- Umbral elegido (acierto objetivo 85,0 %): **0.90** sobre la confianza calibrada: muestra el 43,0 % y acierta el 87,1 % de lo mostrado. **Optimista:** el umbral se eligió mirando estas mismas predicciones.
- Sin mirar a la persona (umbral elegido con las otras dos, medido en ella): muestra el 45,9 % y acierta el 83,5 % (entre personas: 75,9-93,1 %). Umbrales elegidos: {'P1': 0.8, 'P2': 0.85, 'P3': 0.95}.
- Umbral de la app (0.75), por persona: acierto de lo mostrado P1 72,7 % (40,3 % mostrado), P2 81,0 % (72,6 % mostrado), P3 86,9 % (69,5 % mostrado); media entre personas 80,2 ± 7,1.

| Umbral | Mostradas sin calibrar | Acierto | Mostradas calibrado | Acierto |
|---|---|---|---|---|
| 0.30 | 93,4 | 66,8 | 93,3 | 66,9 |
| 0.35 | 93,2 | 66,9 | 92,5 | 67,1 |
| 0.40 | 92,7 | 67,1 | 91,8 | 67,2 |
| 0.45 | 91,3 | 67,0 | 89,5 | 67,7 |
| 0.50 | 89,6 | 67,5 | 85,7 | 68,5 |
| 0.55 | 85,0 | 69,2 | 79,2 | 71,8 |
| 0.60 | 78,8 | 72,5 | 73,2 | 75,6 |
| 0.65 | 74,8 | 75,4 | 68,2 | 78,8 |
| 0.70 | 70,9 | 77,1 | 64,9 | 80,0 |
| 0.75 | 66,7 | 79,4 | 60,8 | 81,4 |
| 0.80 | 63,8 | 80,6 | 56,7 | 83,2 |
| 0.85 | 59,9 | 82,1 | 51,7 | 85,0 |
| 0.90 | 54,1 | 83,7 | 43,0 | 87,1 |
| 0.95 | 43,7 | 87,5 | 27,7 | 91,3 |

## 8. Robustez por condición (sesión 2)

Fuente: `resultados_robustez\2026-11-12_p1p2p3_5semillas_pareado`.

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

## 9. Consistencia estadística σ

Fuente: `resultados_consistencia\2026-10-18_p1p2p3`.

Generado por `scripts/consistencia.py`. No editar a mano.

449 tomas del teléfono. σ_i = promedio sobre las clases de la desviación estándar sesgada de las probabilidades de los vecinos (N_i incluye a la muestra; los vecinos se buscan en todas las tomas, de las dos sesiones); σ de un grupo = promedio de σ_i de sus tomas con al menos otro vecino. ε = percentil de las distancias entre pares de la sesión 1: p0.5 = 0,72, p1 = 0,99, p2 = 1,25, p5 = 1,70, p10 = 2,19. Tabla principal con p0.5: el ε más grande con al menos 90,0 % de vecinos de la misma seña (con ε más grande entran señas distintas y σ mide cuánto se mezclan las clases, no la estabilidad).

Ojo: un modelo que reparte la probabilidad por igual entre todas las clases tiene σ bajo sin acertar. σ se lee siempre junto al acierto.

σ va de 0 (vecinos con las mismas probabilidades) a 0,5 (vecinos que se reparten entre dos clases con certeza); más bajo es más consistente.

## Modelo publicado en la app (temperatura incluida)

| Datos | Tomas | Acierto (%) | σ | Con vecinos | σ_i aciertos | σ_i errores |
|---|---|---|---|---|---|---|
| sesión 1 | 318 | 95,3 | 0,008 | 135 | 0,007 | 0,028 |
| poca luz | 36 | 83,3 | 0,024 | 15 | 0,019 | 0,052 |
| cerca | 36 | 83,3 | 0,008 | 11 | 0,008 | — |
| lejos | 24 | 79,2 | 0,028 | 14 | 0,028 | 0,026 |
| lejos + poca luz | 12 | 75,0 | 0,036 | 6 | 0,033 | 0,053 |
| girado 30° | 23 | 56,5 | 0,000 | 1 | 0,000 | — |

La app ya vio la sesión 1 de estas personas: su σ ahí es optimista.

## Persona no vista (cada persona con un modelo entrenado sin ella)

Semillas: [0, 1, 2]. σ con p0.5, media ± desvío entre semillas.

| Datos | Acierto (%) | σ |
|---|---|---|
| sesión 1 | 67,2 | 0,022 ± 0,003 |
| poca luz | 57,4 | 0,040 ± 0,011 |
| cerca | 64,8 | 0,029 ± 0,007 |
| lejos | 55,6 | 0,038 ± 0,001 |
| lejos + poca luz | 63,9 | 0,043 ± 0,007 |
| girado 30° | 40,6 | 0,004 ± 0,003 |

## Sensibilidad a ε (modelo publicado, sesión 1)

| ε | Vecinos de la misma seña (%) | σ | Con vecinos | Vecinos por toma |
|---|---|---|---|---|
| p0.5 = 0,72 | 98,4 | 0,008 | 135 | 5,2 |
| p1 = 0,99 | 85,9 | 0,023 | 200 | 7,3 |
| p2 = 1,25 | 67,7 | 0,045 | 231 | 12,5 |
| p5 = 1,70 | 43,8 | 0,086 | 245 | 30,5 |
| p10 = 2,19 | 30,5 | 0,127 | 253 | 62,1 |

## Tabla en LaTeX (booktabs)

```latex
\begin{tabular}{lrrrr}
\toprule
Datos & Acierto app (\%) & $\sigma$ app & Acierto nueva (\%) & $\sigma$ nueva \\
\midrule
sesión 1 & 95,3 & 0,008 & 67,2 & 0,022 \\
poca luz & 83,3 & 0,024 & 57,4 & 0,040 \\
cerca & 83,3 & 0,008 & 64,8 & 0,029 \\
lejos & 79,2 & 0,028 & 55,6 & 0,038 \\
lejos + poca luz & 75,0 & 0,036 & 63,9 & 0,043 \\
girado 30° & 56,5 & 0,000 & 40,6 & 0,004 \\
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
