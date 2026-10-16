# Calibración sobre personas no vistas

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
