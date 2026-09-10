| Variante | Tamano | vs f32 | Accuracy | F1-macro | Latencia mediana | p95 | Predicciones distintas |
|---|---|---|---|---|---|---|---|
| float32 | 648 KB | 1.00x | 0.9620 | 0.9633 | 0.78 ms | 1.07 ms | 0.0 % |
| dinamica (int8 pesos) | 182 KB | 0.28x | 0.9620 | 0.9633 | 1.15 ms | 1.54 ms | 0.0 % |
| entera (int8 + calibracion) | 179 KB | 0.28x | 0.9565 | 0.9580 | 21.88 ms | 23.05 ms | 1.1 % |
