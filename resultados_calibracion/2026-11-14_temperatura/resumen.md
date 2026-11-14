# Temperatura de calibración: con qué datos se ajusta y cuánto importa (auditoría, ML-M7)

Generado el 2026-11-14 con `scripts/temperatura_lopo.py` (las tablas completas están en `tablas.md`
y todos los números en `resultados.json`). No toca `calibracion_app.json` ni la app.

## Qué se midió

Datos: 1409 tomas, 13 clases (P1: 1091 de webcam + 106 de teléfono; P2 y P3: 106 de teléfono cada
una). Evaluación: dejar una persona afuera (LOPO) × 3 semillas = 9 pliegues por corrida; en cada
pliegue se mide sobre las 106 tomas de teléfono de la persona que quedó afuera (318 predicciones por
persona, 954 en total). Dos corridas:

- **2026-11-10_lopo**: las predicciones que ya había guardado `calibrar.py`, reanalizadas sin entrenar.
- **2026-11-14 (reentrenada)**: la misma receta (`calibrar.py`: mismo split, mismo `entrenar`, GPU),
  guardando además las predicciones del 20 % de calibración. Va en `entrenada/`.

La corrida 2026-10-16_lopo, de la que sale la T publicada (1,2008), no guardó predicciones: solo
se pueden comparar sus T por pliegue (1,00 a 1,52).

## (1) Con qué datos se ajustó la T publicada

`scripts/calibrar.py`, líneas 147-151: para cada persona p, `otros = persona != p`; de `otros` se
reserva un 20 % estratificado (`cal`), se entrena con el 80 % restante y T se ajusta minimizando la
log-verosimilitud sobre `cal`. La T publicada es la mediana de los 9 pliegues.

Ese 20 % no lo vio el modelo y no tiene a p, pero **sus personas sí están en el entrenamiento del
pliegue**: son tomas nuevas de personas conocidas, no personas nuevas. Además, salvo cuando p = P1,
lo domina la webcam de P1:

| Pliegue (persona afuera) | Tomas en `cal` | Webcam (P1) | Teléfono |
|---|---|---|---|
| P1 | 43 | 0 | 43 (P2 + P3) |
| P2 o P3 | 261 | 218-223 | 38-43 |

Sobre la persona no vista solo se **evalúa** (ECE, log-loss, umbral). La frase «ajustada sobre
personas no vistas» describe mal el procedimiento: la T se ajustó con tomas reservadas de las
personas del entrenamiento y se evaluó sobre la persona no vista.

## (2) Cuánto cambia T según con qué se ajuste

| Cómo se ajusta T | 2026-11-10 (mediana, rango de 9 pliegues) | 2026-11-14 (mediana, rango) |
|---|---|---|
| Receta: 20 % reservado (personas vistas) | 1,29 (0,99-1,67) | 1,21 (0,91-1,53) |
| 20 % reservado, solo tomas de teléfono | — (no hay predicciones) | 1,39 (1,03-2,08) |
| **Anidada**: predicciones LOPO de las otras dos personas, misma semilla | **2,00 (1,83-2,13)** | **1,90 (1,62-2,10)** |
| Oráculo: la misma persona que se evalúa (cota, no desplegable) | 2,02 | 1,99 |
| Una sola T con todas las predicciones LOPO | 1,98 | 1,89 |

En los 18 pliegues (9 por corrida), la T anidada es mayor que la de la receta. La T publicada
(1,20) queda por debajo del menor valor anidado (1,62).

## (3) Efecto, por persona (umbral de la app 0,75; las 3 semillas juntas)

Corrida 2026-11-14 (reentrenada). Entre paréntesis, el rango entre semillas. «Confianza» es la
confianza media de lo que la app mostraría, el «xx %» que ve la persona. La T no cambia la clase
predicha: el accuracy es el mismo con cualquier T (P1 50,6 %, P2 71,7 %, P3 72,6 %).

| Persona | T | ECE % | Log-loss | Mostradas % | Acierto de lo mostrado % | Confianza % |
|---|---|---|---|---|---|---|
| P1 | publicada 1,20 | 19,7 | 1,626 | 38,1 (23,6-45,3) | 78,5 (75,0-84,0) | 88,7 |
| P1 | anidada | 10,9 | 1,463 | 15,1 (0,0-25,5) | 89,6 (81,5-100,0) | 81,1 |
| P2 | publicada 1,20 | 16,2 | 1,073 | 68,9 (63,2-72,6) | 80,8 (70,7-89,6) | 93,1 |
| P2 | anidada | 9,9 | 0,971 | 41,8 (29,2-48,1) | 86,5 (78,4-100,0) | 87,3 |
| P3 | publicada 1,20 | 14,5 | 1,197 | 66,4 (60,4-77,4) | 82,9 (81,5-85,9) | 94,0 |
| P3 | anidada | 7,2 | 1,032 | 43,7 (38,7-48,1) | 87,1 (84,3-91,5) | 88,7 |
| Todas | sin calibrar | 18,6 | 1,445 | 63,2 | 78,8 | 94,2 |
| Todas | publicada 1,20 | 14,8 | 1,299 | 57,8 | 81,1 | 92,5 |
| Todas | solo teléfono | 10,6 | 1,246 | 49,9 | 84,0 | 90,5 |
| Todas | anidada | 4,1 | 1,155 | 33,5 | 87,2 | 87,0 |

La corrida 2026-11-10 da lo mismo. Todas: con T publicada, ECE 15,3 %, log-loss 1,302, se muestra el
62,3 % y acierta el 81,3 % (confianza 93,3 %); con la anidada, ECE 5,4 %, log-loss 1,126, se muestra el
36,7 % y acierta el 88,0 % (confianza 86,8 %). Por persona, P1 baja de 39,0 % a 14,2 % mostrado.

**A igual cobertura, la T casi no cambia el acierto.** La curva mostradas/acierto es la misma con
otro umbral. En 2026-11-14, la T publicada con umbral 0,75 muestra el 57,8 % y acierta el 81,1 %; la
anidada con 0,55, el 58,0 % y el 81,2 %. En 2026-11-10: 62,3/81,3 contra 61,8/81,5. Cambiar T sin
cambiar el umbral equivale a subir el umbral: con T ≈ 1,9, el umbral 0,75 es más estricto que el
0,90 de hoy (T 1,20 a 0,90: 39,7 % mostrado y 86,5 % de acierto).

## Conclusión (para citar)

La temperatura de la app (T = 1,20) se ajustó sobre tomas reservadas de las personas del
entrenamiento, dominadas por la webcam, y no sobre personas nuevas. Se compararon dos corridas LOPO
con 3 personas, 3 semillas y 106 tomas de teléfono por persona (954 predicciones por corrida). Si T
se ajusta con predicciones de personas no vistas por el modelo, sin la persona evaluada, sale entre
1,62 y 2,13 (mediana 1,90-2,00): mayor que la de la receta en los 18 pliegues. Con esa T, el ECE
baja de 14,8-15,3 % a 4,1-5,4 % y la log-loss de 1,30 a 1,13-1,16. Con T = 1,20, la confianza que se
muestra sobreestima el acierto en unos 11-12 puntos (92,5-93,3 % contra 81,1-81,3 %).

Lo que se puede lograr entre cobertura y acierto no cambia: el umbral 0,75 con T = 1,20 equivale a
0,55 con la T anidada.

## Límites

- Tres personas; una de ellas (P1) es también la de la webcam. Con 106 tomas por persona, el ECE de
  15 bins es ruidoso; la log-loss es más estable. Los rangos entre semillas son amplios
  (P1 anidada: entre 0 y 25,5 % mostrado según la semilla).
- La T se aplica en la app a otro modelo (`runs/2026-10-16_final`, entrenado con las 3 personas). Para
  ese modelo no hay persona no vista: su calibración con una persona nueva no se puede medir con
  estos datos.
- El entrenamiento en GPU no es determinista: la receta dio 1,20 (2026-10-16), 1,29 (2026-11-10) y
  1,21 (2026-11-14).
