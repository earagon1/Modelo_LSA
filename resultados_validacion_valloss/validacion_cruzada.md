Validacion cruzada estratificada, 5 folds. Early stopping sobre `val_loss`.

| Metrica | Media entre folds | Out-of-fold |
|---|---|---|
| Accuracy | 0.9457 +/- 0.0154 | 0.9457 |
| F1-macro | 0.9457 +/- 0.0143 | 0.9460 |
| F1-micro | 0.9457 +/- 0.0154 | 0.9457 |
| sigma (consistencia) | - | 0.1049 |

| Fold | Muestras | Epocas | Accuracy | F1-macro |
|---|---|---|---|---|
| 1 | 184 | 78 | 0.9402 | 0.9392 |
| 2 | 184 | 61 | 0.9293 | 0.9311 |
| 3 | 184 | 66 | 0.9728 | 0.9712 |
| 4 | 184 | 67 | 0.9511 | 0.9509 |
| 5 | 184 | 42 | 0.9348 | 0.9362 |
