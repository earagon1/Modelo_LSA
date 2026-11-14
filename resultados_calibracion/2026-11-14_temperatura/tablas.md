# Temperatura de calibración: con qué datos se ajusta y cuánto importa (ML-M7)

Umbral de la app: 0,75. T publicada en la app: 1,2008. Generado por `scripts/temperatura_lopo.py`.

## Corrida: 2026-11-10_lopo (reanálisis, sin entrenar)

Predicciones guardadas por calibrar.py en `resultados_calibracion/2026-11-10_lopo`. Las T de la receta salen de su resultados.json. 954 predicciones = 3 personas × 3 semillas.

### Temperatura por pliegue (semilla-persona evaluada)

| Pliegue | receta_pliegue | anidada | oraculo |
|---|---|---|---|
| 0-P1 | 1,06 | 1,83 | 2,06 |
| 0-P2 | 1,48 | 2,00 | 1,71 |
| 0-P3 | 1,29 | 1,88 | 1,94 |
| 1-P1 | 1,07 | 2,09 | 2,09 |
| 1-P2 | 1,67 | 2,05 | 2,15 |
| 1-P3 | 1,44 | 2,13 | 2,02 |
| 2-P1 | 1,31 | 1,86 | 2,11 |
| 2-P2 | 0,99 | 1,91 | 1,99 |
| 2-P3 | 1,05 | 2,04 | 1,73 |
| mediana | 1,29 | 2,00 | 2,02 |

Fijas: sin = 1,000, publicada = 1,201, receta_mediana = 1,287. Una sola T ajustada con todas las predicciones LOPO: 1,976.

### Efecto por persona (las semillas juntas)

Mostradas y acierto de lo mostrado con el umbral de la app; entre paréntesis, el rango entre semillas. La argmax no cambia con T, así que el accuracy es el mismo en todos los esquemas.

| Persona | Esquema | ECE % | Log-loss | Mostradas % | Acierto mostradas % | Confianza media de lo mostrado % |
|---|---|---|---|---|---|---|
| P1 (n=318) | sin | 23,6 | 1,906 | 45,9 (39,6-57,5) | 70,5 (64,3-83,7) | 91,5 |
| P1 (n=318) | publicada | 18,6 | 1,719 | 39,0 (33,0-48,1) | 75,0 (68,6-86,8) | 89,7 |
| P1 (n=318) | receta_mediana | 16,4 | 1,664 | 36,5 (31,1-43,4) | 76,7 (71,7-86,5) | 88,7 |
| P1 (n=318) | receta_pliegue | 19,5 | 1,771 | 40,3 (37,7-43,4) | 72,7 (62,5-83,3) | 90,4 |
| P1 (n=318) | anidada | 7,9 | 1,504 | 14,2 (0,0-30,2) | 88,9 (84,4-100,0) | 82,7 |
| P1 (n=318) | oraculo | 8,8 | 1,498 | 10,1 (0,0-20,8) | 90,6 (86,4-100,0) | 79,1 |
| P2 (n=318) | sin | 17,5 | 1,231 | 78,9 (73,6-82,1) | 79,3 (75,9-85,9) | 96,5 |
| P2 (n=318) | publicada | 14,6 | 1,082 | 76,4 (70,8-79,2) | 79,4 (76,2-85,3) | 95,0 |
| P2 (n=318) | receta_mediana | 13,8 | 1,037 | 74,5 (68,9-77,4) | 79,7 (75,6-84,9) | 94,4 |
| P2 (n=318) | receta_pliegue | 11,1 | 1,047 | 72,6 (67,0-81,1) | 81,0 (76,7-85,9) | 93,7 |
| P2 (n=318) | anidada | 8,2 | 0,911 | 56,9 (54,7-60,4) | 84,0 (81,4-87,9) | 87,5 |
| P2 (n=318) | oraculo | 7,9 | 0,906 | 56,3 (51,9-61,3) | 83,8 (81,4-87,7) | 88,3 |
| P3 (n=318) | sin | 17,6 | 1,245 | 75,2 (71,7-79,2) | 84,9 (81,0-88,2) | 95,3 |
| P3 (n=318) | publicada | 13,0 | 1,104 | 71,4 (67,9-74,5) | 86,8 (82,9-91,7) | 93,4 |
| P3 (n=318) | receta_mediana | 11,5 | 1,064 | 69,5 (66,0-73,6) | 87,8 (83,6-92,9) | 92,6 |
| P3 (n=318) | receta_pliegue | 11,9 | 1,075 | 69,5 (68,9-70,8) | 86,9 (83,6-89,3) | 92,8 |
| P3 (n=318) | anidada | 10,7 | 0,963 | 39,0 (32,1-43,4) | 93,5 (88,2-97,7) | 87,3 |
| P3 (n=318) | oraculo | 9,2 | 0,956 | 45,9 (31,1-55,7) | 91,8 (87,9-94,9) | 88,1 |
| todas (n=954) | sin | 18,6 | 1,461 | 66,7 | 79,4 | 94,9 |
| todas (n=954) | publicada | 15,3 | 1,302 | 62,3 | 81,3 | 93,3 |
| todas (n=954) | receta_mediana | 13,5 | 1,255 | 60,2 | 82,2 | 92,5 |
| todas (n=954) | receta_pliegue | 13,8 | 1,298 | 60,8 | 81,4 | 92,6 |
| todas (n=954) | anidada | 5,4 | 1,126 | 36,7 | 88,0 | 86,8 |
| todas (n=954) | oraculo | 4,9 | 1,120 | 37,4 | 87,7 | 87,4 |

### Qué hay en el 20 % de calibración de cada pliegue

| Pliegue | n | webcam | teléfono | por persona |
|---|---|---|---|---|
| 0-P1 | 43 | 0 | 43 | {'P2': 26, 'P3': 17} |
| 0-P2 | 261 | 218 | 43 | {'P1': 241, 'P3': 20} |
| 0-P3 | 261 | 218 | 43 | {'P1': 241, 'P2': 20} |
| 1-P1 | 43 | 0 | 43 | {'P2': 27, 'P3': 16} |
| 1-P2 | 261 | 222 | 39 | {'P1': 240, 'P3': 21} |
| 1-P3 | 261 | 222 | 39 | {'P1': 240, 'P2': 21} |
| 2-P1 | 43 | 0 | 43 | {'P2': 21, 'P3': 22} |
| 2-P2 | 261 | 223 | 38 | {'P1': 244, 'P3': 17} |
| 2-P3 | 261 | 223 | 38 | {'P1': 244, 'P2': 17} |

## Corrida: resultados_calibracion/2026-11-14_temperatura/entrenada (reentrenada con la receta de calibrar.py)

Misma receta que calibrar.py (LOPO, 20 % de calibración estratificado, mismo `entrenar`, GPU), con las predicciones del set de calibración guardadas para ajustar también la T solo con sus tomas de teléfono (`cal_telefono`). 954 predicciones = 3 personas × 3 semillas.

### Temperatura por pliegue (semilla-persona evaluada)

| Pliegue | receta_pliegue | cal_telefono | anidada | oraculo |
|---|---|---|---|---|
| 0-P1 | 1,09 | 1,09 | 1,84 | 1,87 |
| 0-P2 | 1,33 | 1,81 | 1,93 | 1,70 |
| 0-P3 | 1,21 | 1,39 | 1,78 | 1,99 |
| 1-P1 | 1,23 | 1,23 | 2,10 | 2,00 |
| 1-P2 | 1,53 | 2,08 | 2,02 | 2,16 |
| 1-P3 | 1,27 | 1,48 | 2,08 | 2,05 |
| 2-P1 | 1,03 | 1,03 | 1,62 | 2,04 |
| 2-P2 | 0,91 | 1,27 | 1,90 | 1,44 |
| 2-P3 | 1,06 | 1,53 | 1,73 | 1,78 |
| mediana | 1,21 | 1,39 | 1,90 | 1,99 |

Fijas: sin = 1,000, publicada = 1,201, receta_mediana = 1,211. Una sola T ajustada con todas las predicciones LOPO: 1,891.

### Efecto por persona (las semillas juntas)

Mostradas y acierto de lo mostrado con el umbral de la app; entre paréntesis, el rango entre semillas. La argmax no cambia con T, así que el accuracy es el mismo en todos los esquemas.

| Persona | Esquema | ECE % | Log-loss | Mostradas % | Acierto mostradas % | Confianza media de lo mostrado % |
|---|---|---|---|---|---|---|
| P1 (n=318) | sin | 24,0 | 1,792 | 43,1 (32,1-49,1) | 77,4 (73,1-82,4) | 91,5 |
| P1 (n=318) | publicada | 19,7 | 1,626 | 38,1 (23,6-45,3) | 78,5 (75,0-84,0) | 88,7 |
| P1 (n=318) | receta_mediana | 19,5 | 1,619 | 38,1 (23,6-45,3) | 78,5 (75,0-84,0) | 88,5 |
| P1 (n=318) | receta_pliegue | 20,6 | 1,692 | 38,7 (22,6-47,2) | 78,9 (74,0-87,5) | 90,9 |
| P1 (n=318) | cal_telefono | 20,6 | 1,692 | 38,7 (22,6-47,2) | 78,9 (74,0-87,5) | 90,9 |
| P1 (n=318) | anidada | 10,9 | 1,463 | 15,1 (0,0-25,5) | 89,6 (81,5-100,0) | 81,1 |
| P1 (n=318) | oraculo | 11,3 | 1,451 | 8,8 (0,0-18,9) | 100,0 (100,0-100,0) | 78,9 |
| P2 (n=318) | sin | 16,7 | 1,193 | 73,3 (67,9-76,4) | 78,1 (68,8-87,5) | 94,9 |
| P2 (n=318) | publicada | 16,2 | 1,073 | 68,9 (63,2-72,6) | 80,8 (70,7-89,6) | 93,1 |
| P2 (n=318) | receta_mediana | 15,9 | 1,068 | 68,6 (62,3-72,6) | 81,2 (70,7-90,9) | 93,0 |
| P2 (n=318) | receta_pliegue | 12,3 | 1,035 | 67,0 (64,2-68,9) | 81,2 (72,1-87,7) | 93,0 |
| P2 (n=318) | cal_telefono | 8,6 | 0,956 | 51,6 (44,3-60,4) | 86,6 (83,0-90,6) | 88,9 |
| P2 (n=318) | anidada | 9,9 | 0,971 | 41,8 (29,2-48,1) | 86,5 (78,4-100,0) | 87,3 |
| P2 (n=318) | oraculo | 7,4 | 0,952 | 47,5 (44,3-53,8) | 86,1 (83,0-91,5) | 89,0 |
| P3 (n=318) | sin | 17,1 | 1,350 | 73,3 (67,0-84,0) | 80,3 (77,5-85,9) | 95,0 |
| P3 (n=318) | publicada | 14,5 | 1,197 | 66,4 (60,4-77,4) | 82,9 (81,5-85,9) | 94,0 |
| P3 (n=318) | receta_mediana | 14,4 | 1,191 | 66,0 (60,4-77,4) | 83,3 (81,7-85,9) | 94,0 |
| P3 (n=318) | receta_pliegue | 15,5 | 1,207 | 68,6 (60,4-76,4) | 81,7 (78,1-85,9) | 93,9 |
| P3 (n=318) | cal_telefono | 9,3 | 1,091 | 59,4 (50,9-69,8) | 85,2 (83,8-87,0) | 91,5 |
| P3 (n=318) | anidada | 7,2 | 1,032 | 43,7 (38,7-48,1) | 87,1 (84,3-91,5) | 88,7 |
| P3 (n=318) | oraculo | 8,5 | 1,029 | 42,5 (33,0-48,1) | 88,1 (85,7-92,2) | 87,8 |
| todas (n=954) | sin | 18,6 | 1,445 | 63,2 | 78,8 | 94,2 |
| todas (n=954) | publicada | 14,8 | 1,299 | 57,8 | 81,1 | 92,5 |
| todas (n=954) | receta_mediana | 14,6 | 1,293 | 57,5 | 81,4 | 92,4 |
| todas (n=954) | receta_pliegue | 14,9 | 1,312 | 58,1 | 80,9 | 92,9 |
| todas (n=954) | cal_telefono | 10,6 | 1,246 | 49,9 | 84,0 | 90,5 |
| todas (n=954) | anidada | 4,1 | 1,155 | 33,5 | 87,2 | 87,0 |
| todas (n=954) | oraculo | 5,2 | 1,144 | 32,9 | 88,2 | 87,6 |

### Qué hay en el 20 % de calibración de cada pliegue

| Pliegue | n | webcam | teléfono | por persona |
|---|---|---|---|---|
| 0-P1 | 43 | 0 | 43 | {'P2': 26, 'P3': 17} |
| 0-P2 | 261 | 218 | 43 | {'P1': 241, 'P3': 20} |
| 0-P3 | 261 | 218 | 43 | {'P1': 241, 'P2': 20} |
| 1-P1 | 43 | 0 | 43 | {'P2': 27, 'P3': 16} |
| 1-P2 | 261 | 222 | 39 | {'P1': 240, 'P3': 21} |
| 1-P3 | 261 | 222 | 39 | {'P1': 240, 'P2': 21} |
| 2-P1 | 43 | 0 | 43 | {'P2': 21, 'P3': 22} |
| 2-P2 | 261 | 223 | 38 | {'P1': 244, 'P3': 17} |
| 2-P3 | 261 | 223 | 38 | {'P1': 244, 'P2': 17} |

