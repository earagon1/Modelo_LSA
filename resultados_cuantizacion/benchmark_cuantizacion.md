| Variante | Tamano | vs f32 | Accuracy | F1-macro | Latencia mediana | p95 | Predicciones distintas |
|---|---|---|---|---|---|---|---|
| float32 | 648 KB | 1.00x | 0.9635 | 0.9636 | 0.95 ms | 1.57 ms | 0.0 % |
| dinamica (int8 pesos) | 182 KB | 0.28x | 0.9635 | 0.9636 | 1.22 ms | 1.97 ms | 0.0 % |
| entera (int8 + calibracion) | 179 KB | 0.28x | 0.9635 | 0.9636 | 16.27 ms | 25.02 ms | 1.0 % |
