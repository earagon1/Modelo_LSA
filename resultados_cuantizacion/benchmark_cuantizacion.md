| Variante | Tamano | vs f32 | Accuracy | F1-macro | Latencia mediana | p95 | Predicciones distintas |
|---|---|---|---|---|---|---|---|
| float32 | 648 KB | 1.00x | 0.9728 | 0.9742 | 0.77 ms | 1.20 ms | 0.0 % |
| dinamica (int8 pesos) | 182 KB | 0.28x | 0.9728 | 0.9742 | 1.14 ms | 1.57 ms | 0.0 % |
| entera (int8 + calibracion) | 179 KB | 0.28x | 0.9728 | 0.9742 | 21.98 ms | 25.63 ms | 1.1 % |
