| Variante | Tamano | vs f32 | Accuracy | F1-macro | Latencia mediana | p95 | Predicciones distintas |
|---|---|---|---|---|---|---|---|
| float32 | 648 KB | 1.00x | 0.9667 | 0.9665 | 0.75 ms | 0.96 ms | 0.0 % |
| dinamica (int8 pesos) | 182 KB | 0.28x | 0.9667 | 0.9665 | 1.11 ms | 1.25 ms | 0.0 % |
| entera (int8 + calibracion) | 179 KB | 0.28x | 0.9667 | 0.9666 | 15.82 ms | 21.81 ms | 1.2 % |
