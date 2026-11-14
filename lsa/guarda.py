"""Guarda de los scripts heredados que pisan o borran datos (auditoría, ML-M11).

Los scripts de la raíz (`normalize_samples.py`, `create_keypoints.py`,
`training_model.py`, `export_tflite.py`) son del primer pipeline y escriben sobre
lo que ya existe: los frames de las tomas, los `.h5`, `models/` y `words.json`.
Nada de eso está en git, así que un `python training_model.py` por costumbre se
lleva el modelo o los datos sin vuelta atrás. Antes la única defensa era una
advertencia en el README.

Con esta guarda el script corta antes de tocar nada si hay algo que perder, dice
qué es y qué usar en su lugar, y solo sigue con la bandera `--pisar`. No pregunta
por teclado a propósito: una pregunta se contesta con Enter sin leer, y deja
colgado al que lo corre desde otro programa.

Sin dependencias pesadas, para poder probarla sin TensorFlow.
"""
import os
import sys

BANDERA = "--pisar"


def existentes(rutas):
    """Las rutas que hoy existen, en el orden dado: es lo que el script pisaría."""
    return [r for r in rutas if os.path.exists(r)]


def mensaje(script, en_riesgo, en_su_lugar, maximo=8):
    """El texto con el que corta la guarda (aparte, para poder leerlo en un test)."""
    lineas = [f"{script} no corrió: pisa o borra datos que no están en git.", "", "Tocaría:"]
    lineas += [f"  - {r}" for r in en_riesgo[:maximo]]
    if len(en_riesgo) > maximo:
        lineas.append(f"  ... y {len(en_riesgo) - maximo} más")
    lineas += ["", f"En su lugar: {en_su_lugar}",
               f"Si de verdad hay que correrlo, respaldar eso primero y repetir con {BANDERA}."]
    return "\n".join(lineas)


def exigir_permiso(script, en_riesgo, en_su_lugar, argv=None):
    """Corta con SystemExit si hay algo en riesgo y no vino la bandera.

    Si no hay nada en riesgo no pide nada: el script no puede romper lo que no existe.
    """
    argv = sys.argv[1:] if argv is None else argv
    if en_riesgo and BANDERA not in argv:
        raise SystemExit(mensaje(script, list(en_riesgo), en_su_lugar))


def clases_sin_tomas(clases, etiquetas):
    """Las clases de `clases` que no aparecen en `etiquetas`, en el orden de `clases`.

    Entrenar con una clase sin tomas no falla: deja una salida del modelo que nunca
    se entrenó, y `words.json` la sigue anunciando. Es lo que pasa cuando a una
    clase le falta su `.h5`.
    """
    presentes = set(etiquetas)
    return [c for c in clases if c not in presentes]
