# Calibración sobre personas no vistas

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
