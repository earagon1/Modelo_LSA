# Consistencia estadística σ (propuesta, sección de evaluación)

Generado por `scripts/consistencia.py`. No editar a mano.

449 tomas del teléfono. σ_i = promedio sobre las clases de la desviación estándar sesgada de las probabilidades de los vecinos (N_i incluye a la muestra; los vecinos se buscan en todas las tomas, de las dos sesiones); σ de un grupo = promedio de σ_i de sus tomas con al menos otro vecino. ε = percentil de las distancias entre pares de la sesión 1: p0.5 = 0,72, p1 = 0,99, p2 = 1,25, p5 = 1,70, p10 = 2,19. Tabla principal con p0.5: el ε más grande con al menos 90,0 % de vecinos de la misma seña (con ε más grande entran señas distintas y σ mide cuánto se mezclan las clases, no la estabilidad).

Ojo: un modelo que reparte la probabilidad por igual entre todas las clases tiene σ bajo sin acertar. σ se lee siempre junto al acierto.

σ va de 0 (vecinos con las mismas probabilidades) a 0,5 (vecinos que se reparten entre dos clases con certeza); más bajo es más consistente.

## Modelo publicado en la app (temperatura incluida)

| Datos | Tomas | Acierto (%) | σ | Con vecinos | σ_i aciertos | σ_i errores |
|---|---|---|---|---|---|---|
| sesión 1 | 318 | 95,3 | 0,008 | 135 | 0,007 | 0,028 |
| poca luz | 36 | 83,3 | 0,024 | 15 | 0,019 | 0,052 |
| cerca | 36 | 83,3 | 0,008 | 11 | 0,008 | — |
| lejos | 24 | 79,2 | 0,028 | 14 | 0,028 | 0,026 |
| lejos + poca luz | 12 | 75,0 | 0,036 | 6 | 0,033 | 0,053 |
| girado 30° | 23 | 56,5 | 0,000 | 1 | 0,000 | — |

La app ya vio la sesión 1 de estas personas: su σ ahí es optimista.

## Persona no vista (cada persona con un modelo entrenado sin ella)

Semillas: [0, 1, 2]. σ con p0.5, media ± desvío entre semillas.

| Datos | Acierto (%) | σ |
|---|---|---|
| sesión 1 | 67,2 | 0,022 ± 0,003 |
| poca luz | 57,4 | 0,040 ± 0,011 |
| cerca | 64,8 | 0,029 ± 0,007 |
| lejos | 55,6 | 0,038 ± 0,001 |
| lejos + poca luz | 63,9 | 0,043 ± 0,007 |
| girado 30° | 40,6 | 0,004 ± 0,003 |

## Sensibilidad a ε (modelo publicado, sesión 1)

| ε | Vecinos de la misma seña (%) | σ | Con vecinos | Vecinos por toma |
|---|---|---|---|---|
| p0.5 = 0,72 | 98,4 | 0,008 | 135 | 5,2 |
| p1 = 0,99 | 85,9 | 0,023 | 200 | 7,3 |
| p2 = 1,25 | 67,7 | 0,045 | 231 | 12,5 |
| p5 = 1,70 | 43,8 | 0,086 | 245 | 30,5 |
| p10 = 2,19 | 30,5 | 0,127 | 253 | 62,1 |

## Tabla en LaTeX (booktabs)

```latex
\begin{tabular}{lrrrr}
\toprule
Datos & Acierto app (\%) & $\sigma$ app & Acierto nueva (\%) & $\sigma$ nueva \\
\midrule
sesión 1 & 95,3 & 0,008 & 67,2 & 0,022 \\
poca luz & 83,3 & 0,024 & 57,4 & 0,040 \\
cerca & 83,3 & 0,008 & 64,8 & 0,029 \\
lejos & 79,2 & 0,028 & 55,6 & 0,038 \\
lejos + poca luz & 75,0 & 0,036 & 63,9 & 0,043 \\
girado 30° & 56,5 & 0,000 & 40,6 & 0,004 \\
\bottomrule
\end{tabular}
```
