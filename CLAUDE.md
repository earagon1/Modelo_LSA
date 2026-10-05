# CLAUDE.md: modelo_LSA (entrenamiento de SeÑAR)

Pipeline Python de la tesina SeÑAR (Evelin Aragón, UNGS): captura con webcam,
extracción de landmarks, entrenamiento de la LSTM, export a TFLite para la app
Android, validación y un módulo de aprendizaje federado (`federated/`).

## Dónde está todo
| Qué | Ruta |
|---|---|
| Este repo (`earagon1/Modelo_LSA`, rama `master`) | `C:\Users\Evelin\Desktop\Tesis\LSA-training\modelo_LSA` |
| Ojo: repo de afuera | `LSA-training/` es **otro** repo, casi sin uso, que ve estos mismos archivos. Antes de commitear: `git rev-parse --show-toplevel` tiene que dar `.../modelo_LSA` |
| App Android | `C:\Users\Evelin\Desktop\Tesis\TraductorLSA` (ver su `CLAUDE.md`) |
| Plan y avance | `C:\Users\Evelin\Desktop\Tesis\plan\` (**leer `progreso.md` al empezar**) |
| Respaldos de datos | Disco externo `D:\Respaldo-Tesis\<fecha>\` con `MANIFIESTO.sha256` |

## Entorno
- Conda `tesis3`: Python 3.10.14, TF 2.10.1, mediapipe 0.10.11, protobuf 3.19.6, numpy 1.26.4, sklearn 1.5.2.
- `conda` no está en el PATH de bash. Se usa el intérprete directo: `C:/Users/Evelin/anaconda3/envs/tesis3/python.exe`.
- GPU: RTX 2060. CUDA 11.2 + cuDNN 8.1 van **dentro del env** (`conda-forge`, instalados el 2026-10-05). TF solo las ve si `envs/tesis3/Library/bin` está en el PATH. En Git Bash, con la ruta en formato POSIX (con `C:/` el PATH se corta en los dos puntos):
  `export PATH="/c/Users/Evelin/anaconda3/envs/tesis3/Library/bin:$PATH"`
  En PowerShell: `$env:PATH = "C:\Users\Evelin\anaconda3\envs\tesis3\Library\bin;$env:PATH"`. Sin esto, TF corre en CPU sin avisar más que con un warning.
- Si hay que instalar algo con `conda` en este env, hacerlo desde PowerShell: los scripts post-instalación son `.bat` y fallan desde bash.
- **No instalar ni actualizar paquetes en `tesis3` sin preguntar.** `requirements.txt` está incompleto (faltan sklearn, matplotlib, scipy); el reemplazo es PY-D9.
- `constants.ROOT_PATH = os.getcwd()`: **todo se corre desde la raíz de este repo**.

## Comandos
```bash
PY=C:/Users/Evelin/anaconda3/envs/tesis3/python.exe
$PY verify_dataset_mix.py                   # conteos por fuente y por clase (solo lee)
$PY federated/test_fedavg.py                # 9 tests de corrección del federado
$PY federated/verificar_export.py <json>    # valida un export del teléfono (solo lee)
$PY validacion_cruzada.py                   # 5-fold OOF (lento; escribe en resultados_validacion/)
```

## Datos (gitignoreados, solo existen en esta máquina)
- `frame_actions/<clase>/sample_<timestamp>/frame_NN.jpg`: 12 clases × 100 tomas, webcam, **una sola persona**.
- `data/keypoints/<clase>.h5`: landmarks Holistic (`lh[63] ‖ rh[63]`); 1200 secuencias de 15 frames, con el 33 % de los frames vacíos (ver `plan/progreso.md`).
- `models/`: `actions_15.keras`, `actions_15*.tflite` y `words.json` (el orden **es** el índice de salida).
- El modelo publicado se entrenó **solo con la webcam**. No hay `lsa_samples.json` del teléfono en el repo.

## Lo que NO se hace (pierde datos o rompe el modelo)
- **Nunca correr `normalize_samples.py`**: borra frames con `os.remove`.
- **`create_keypoints.py` sobrescribe todos los `.h5`** y lee los frames sin ordenar (`helpers.py:91`). No correrlo sin un respaldo previo.
- **`training_model.py` y `export_tflite.py` sobrescriben el modelo de producción** (`models/`), y `training_model.py` además **reescribe `models/words.json`** con las etiquetas que encuentre en `lsa_samples.json`. Respaldar `models/` antes.
- **No usar `finetune_from_lsa_json.py`**: arma el label map alfabético y cruza las clases.
- **No dejar la salida de `h5_to_json.py` como `lsa_samples.json`** en la raíz: duplica muestras entre train y val.
- No borrar ni mover `resultados_*/` ni `federated/resultados_*`: son la evidencia de los informes actuales (dataset viejo de 920). Las corridas nuevas van a directorios nuevos.
- Antes de cualquier tarea que toque datos: nuevo respaldo en `D:\Respaldo-Tesis\<fecha>\` con su manifiesto.

## Convenciones
- **Nombres de clases:** minúsculas, sin tildes, con espacio (`como estas`, `mi nombre es`, `por favor`). Tienen que coincidir: carpeta de `frame_actions/`, `.h5`, `models/words.json`, `TraductorLSA/app/src/main/assets/words.json`, `NOMBRES_PARA_MOSTRAR` (`DictionaryScreen.kt`) y `assets/dictionary/<id con _>.jpg`. Detalle en `ejecucion.txt`.
- **Publicar un modelo en la app** = copiar `.tflite` + `words.json` **juntos** a `TraductorLSA/app/src/main/assets/`. Cuando exista, lo hace solo `scripts/publicar_modelo.py`.
- **Git:** commits directo a `master`, sin ramas, **sin `Co-Authored-By`**, en español y con el *por qué* en el cuerpo. Sin push salvo que lo pida.
- Estilo: comentarios que explican el *por qué*; funciones puras y testeables; los scripts nuevos van en `scripts/` y la lógica, en el paquete `lsa/` (plan, §3.1).

## Contrato de features
El contrato v2 vive en `contract/` (espejado con TraductorLSA, con un test de hash). Hasta que exista, la referencia es `plan/modelo_lsa.md` §1.4 y §2, más los hallazgos medidos en `plan/progreso.md`.
