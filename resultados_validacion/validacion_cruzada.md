Validacion cruzada estratificada, 5 folds. Early stopping sobre `accuracy`.

| Metrica | Media entre folds | Out-of-fold |
|---|---|---|
| Accuracy | 0.9489 +/- 0.0101 | 0.9489 |
| F1-macro | 0.9492 +/- 0.0101 | 0.9498 |
| F1-micro | 0.9489 +/- 0.0101 | 0.9489 |
| sigma (consistencia) | - | 0.1053 |

| Fold | Muestras | Epocas | Accuracy | F1-macro |
|---|---|---|---|---|
| 1 | 184 | 75 | 0.9620 | 0.9605 |
| 2 | 184 | 58 | 0.9511 | 0.9515 |
| 3 | 184 | 47 | 0.9565 | 0.9582 |
| 4 | 184 | 65 | 0.9348 | 0.9335 |
| 5 | 184 | 47 | 0.9402 | 0.9424 |
