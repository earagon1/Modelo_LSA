# Módulo de aprendizaje federado

> Vive dentro del repo **Modelo_LSA** (`earagon1/Modelo_LSA`), junto al resto
> del código de entrenamiento. Las rutas de `data.py` se resuelven relativas
> a la raíz de ese repo, así que mover esta carpeta rompe la carga del
> dataset.

Implementación de FedAvg (McMahan et al., 2017) sobre el clasificador LSTM de
señas, más las métricas de evaluación que la propuesta de tesina compromete y
que no existían en el proyecto: F1-macro, F1-micro, matriz de confusión y la
métrica de consistencia estadística σ.

Corre en el entorno `tesis3` (Python 3.10 + TensorFlow 2.10.1). **No requiere
instalar nada nuevo**: usa numpy, pandas, pytables, scikit-learn y matplotlib,
que ya estaban.

Todos los comandos se ejecutan **desde la raiz de este repo** (`modelo_LSA/`):

```bash
C:/Users/Evelin/anaconda3/envs/tesis3/python.exe federated/test_fedavg.py
```

---

## Archivos

| Archivo | Qué hace |
|---|---|
| `data.py` | Carga los `.h5` de `data/keypoints` y los parte en N clientes (IID, Dirichlet non-IID, o por persona real) |
| `model.py` | La misma red LSTM que el `model.py` de produccion, replicada para no arrastrar `constants.py`/cv2 |
| `fedavg.py` | El algoritmo: entrenamiento local, promediado ponderado, ciclo de rondas |
| `metrics.py` | Accuracy, F1 macro/micro, matriz de confusión y consistencia σ |
| `compresion.py` | Cuantización de las actualizaciones que suben los clientes |
| `test_fedavg.py` | Los nueve tests de corrección |
| `run_experiment.py` | Corre los cuatro escenarios y genera la tabla del informe |
| `run_compresion.py` | Barre bits por valor y mide accuracy contra MB subidos |
| `verificar_export.py` | Revisa un `lsa_samples.json` exportado desde la app |

---

## Los tests de corrección

No miden si el modelo aprende: miden si la implementación hace lo que dice.
Corren en CPU y en segundos.

1. **Promediado.** K clientes con pesos idénticos → el promedio es ese mismo peso.
2. **Ponderación.** Cliente A con 90 muestras y B con 10 → el resultado queda 9
   veces más cerca de A. Verifica que no se esté promediando sin ponderar.
3. **Identidad.** Un solo cliente con *todos* los datos y una época local tiene
   que dar **exactamente** lo mismo que el entrenamiento centralizado de una
   época. Es la prueba más fuerte de que el ciclo de rondas no corrompe nada.
4. **Arquitectura.** La red de `federated/model.py` es idéntica a la del
   `model.py` de produccion. Si alguien toca el original y no replica el
   cambio, este test falla.
5. **Lectura del export.** Un `lsa_samples.json` con el esquema exacto que
   produce la app, incluidas muestras viejas sin `client_id`, se carga y se
   particiona por persona correctamente.
6. **Deltas.** Mandar deltas y sumarlos al global da lo mismo que promediar
   pesos. Es la base algebraica de toda la compresión.
7. **Error acotado.** El redondeo determinista nunca se desvía más de media
   escala.
8. **Insesgadez.** El redondeo estocástico promedia al valor original; el
   determinista no.
9. **Bytes.** El ahorro es proporcional a los bits, sin sorpresas.

### Un hallazgo del test de identidad

La primera versión falló con una diferencia de 5.9e-03, que es demasiado grande
para ser ruido de punto flotante. La causa: al reusar una sola instancia de
modelo para todos los clientes, **los momentos acumulados por Adam se filtraban
de un cliente al siguiente**. El cliente 1 empezaba su entrenamiento
condicionado por el estado que Adam había construido entrenando al cliente 0.

Eso rompe la premisa del esquema federado por dos razones: es una transferencia
de información entre clientes que no pasa por el promediado, y hace que el
resultado dependa del orden en que se recorren los clientes. En FedAvg cada
cliente es *stateless* entre rondas.

Está resuelto en `fedavg.reset_optimizer_state()`, que se llama al inicio de
cada entrenamiento local. **Vale la pena mencionarlo en el informe**: es
exactamente el tipo de error silencioso que los tests de corrección justifican.

---

## El experimento

```bash
python federated/run_experiment.py --repeticiones 5          # recomendado
python federated/run_experiment.py --clients 5 --alpha 0.1 --repeticiones 5
python federated/run_experiment.py --quick                   # prueba de humo
```

### Correr siempre con varias réplicas

**Una sola corrida no sirve para comparar.** El conjunto de test tiene ~184
muestras, así que **una muestra vale 0,54 % de accuracy**, y la curva por ronda
oscila 2–3 puntos entre rondas consecutivas. Una diferencia de 3 puntos entre
dos escenarios son 6 muestras de test: es ruido, no un resultado.

La primera corrida de este módulo lo dejó a la vista: FedAvg le "ganaba" al
centralizado y el non-IID le "ganaba" al IID, las dos cosas al revés de lo
esperado, y las dos dentro del margen de oscilación.

Por eso `--repeticiones N` corre el experimento completo N veces con semillas
distintas y reporta **media ± desvío**. Cada réplica resortea el split
train/test, las particiones y la inicialización, así que el desvío estima la
variabilidad total del procedimiento. Con `--repeticiones 1` la tabla sale con
una advertencia impresa encima.

El script además imprime una **lectura automática** que compara cada escenario
contra el piso y contra el techo con un **t-test pareado** sobre las réplicas.
El test es pareado porque dentro de una réplica los cuatro escenarios comparten
semilla — mismo split, misma partición, misma inicialización — así que el ruido
común se cancela y el test gana mucha potencia. Está para evitar leer como
conclusión algo que no lo es; no reemplaza el análisis del informe.

Si cambiás el criterio de interpretación, `--reanalizar <dir>` rehace la tabla
y la lectura desde el `resultados.json` guardado, sin volver a entrenar.

### Los cuatro escenarios

Cada uno contesta algo distinto:

| Escenario | Qué mide |
|---|---|
| **Centralizado** | Techo. Todos los datos juntos. Es lo que FedAvg intenta igualar sin moverlos |
| **Solo local** | Piso. Un cliente aislado, sin colaborar. **Si FedAvg no le gana a esto, el enfoque no aporta nada** |
| **FedAvg IID** | Reparto uniforme. El caso amable |
| **FedAvg non-IID** | Reparto Dirichlet: cada cliente con un subconjunto sesgado del vocabulario. El caso realista |

Dos condiciones que hacen que la tabla signifique algo:

- **Mismo presupuesto de épocas** para todos (`rondas × épocas_locales`), así
  ninguno gana por entrenar más.
- **Mismo conjunto de test**, estratificado y que ningún cliente ve jamás.

Salidas en el directorio que indique `--salida` (por defecto
`federated/resultados/`):

| Archivo | Contenido |
|---|---|
| `tabla_resultados.md` | La tabla para el informe, más la lectura automática |
| `resultados.json` | Métricas de cada réplica y las curvas completas |
| `curva_accuracy.png` | Accuracy vs. ronda, con banda de desvío entre réplicas |
| `confusion_*.png` | Matriz de confusión por escenario (primera réplica) |

---

## Compresión de las actualizaciones

FedAvg sin comprimir movió **193 MB** en la corrida de 5 clientes por 30 rondas.
Para un escenario móvil eso es prohibitivo, y es exactamente el problema que
ataca Konečný et al. \[3\], que ya está en tu bibliografía.

```bash
python federated/run_compresion.py --repeticiones 5
```

**Qué se manda.** En vez del peso `w_k`, el cliente manda el delta
`d_k = w_k − w_global`. El servidor ya tiene `w_global`, así que reconstruye lo
mismo sumando el promedio de los deltas. Es algebraicamente idéntico —lo
verifica el test 6— pero el delta es chico y está centrado en cero, así que se
cuantiza mucho mejor que el peso absoluto.

**Solo se comprime la subida.** La bajada del modelo global va en float32,
porque en un escenario móvil el cuello de botella es el ancho de banda de
subida. Es el mismo recorte que hace la literatura.

**Por qué el redondeo estocástico.** Al cuantizar hay que decidir cómo
redondear, y no da igual. El determinista introduce un sesgo sistemático que
**no se cancela al promediar entre clientes**: se acumula ronda tras ronda y
desvía el modelo global. El estocástico redondea para arriba o para abajo con
probabilidad proporcional a la distancia, de modo que el valor esperado es el
original; el error queda con media cero y se cancela al promediar. El
experimento compara las dos a igual cantidad de bits, que es la forma de aislar
ese efecto.

**Un límite que hay que declarar en el informe:** los valores se siguen
guardando en float32, no se empaquetan de verdad en 4 u 8 bits. Lo que se
simula es el *costo* de comunicación, calculado analíticamente. Se mide cuánto
se ahorraría, no se implementa el protocolo de transporte.

## Sobre la métrica σ

`metrics.consistency_sigma` implementa al pie el procedimiento de las páginas
5 a 7 de la propuesta: vecindad `N_i = {j : ||X_i − X_j|| < ε}`, desvío
**sesgado** (divide por *m*, no por *m−1*) por clase sobre esa vecindad,
promedio sobre las C clases, y promedio final sobre los *i* con `|N_i| > 1`.

La propuesta no fija el valor de ε porque depende de la escala de los
landmarks. `sugerir_epsilon()` lo deriva de los datos: toma el percentil 5 de
las distancias entre pares distintos, o sea que dos gestos son vecinos si están
entre el 5 % de los pares más cercanos. Es una elección documentada y
reproducible, que es lo que hay que poder justificar en la defensa.

**Cuidado al interpretar σ**: un σ bajo significa estabilidad *sólo si el
modelo además acierta*. Un modelo que predice siempre la misma clase tiene σ ≈ 0
y es inútil. Por eso en la tabla σ va al lado de accuracy y F1, nunca solo.
El script también reporta `n_con_vecindad`: si son pocas muestras, el σ no es
representativo y hay que subir ε.

---

## Lo que falta para que el experimento sea sobre personas y no sintético

Hoy las 920 muestras las grabó una sola persona, así que la partición en
clientes es **sintética**. Funciona para validar el algoritmo, y la partición
Dirichlet es el procedimiento estándar de la literatura, pero no permite
afirmar nada sobre generalización interusuario.

Para que sí lo permita:

1. Grabar 3–5 personas. Una sola sesión de captura cubre a la vez los clientes
   federados y las pruebas de robustez por usuario que la propuesta también
   compromete (sección 3.5).
2. Guardar de quién es cada muestra. En la app el `user.id` de Clerk ya está
   disponible; hay que agregarlo al export de `lsa_samples.json`.
3. Usar `data.partition_by_client()` en lugar de las particiones sintéticas.

Hasta entonces, el informe tiene que decir explícitamente que la partición es
simulada. Es un límite del experimento, no un defecto de la implementación.

---

## Lo que este módulo deliberadamente no hace

- **No entrena dentro del teléfono.** La Fase 4 de la propuesta pide
  "simulación de entrenamiento distribuido en instancias controladas", que es
  esto. El entrenamiento on-device de LSTM con TFLite está mal soportado y es
  un proyecto aparte.
- **No usa TensorFlow Federated.** TFF no publica wheels para Windows y fija
  versiones de TF que romperían el setup de GPU con TF 2.10 + cuDNN 8.9.7.
  FedAvg son sesenta líneas y tenerlas a la vista es preferible a una caja
  negra que además no instala.
- **No implementa secure aggregation ni privacidad diferencial.** Están citados
  en el marco teórico, no comprometidos como entregable.
