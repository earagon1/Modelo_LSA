"""Extractor de landmarks igual al de la app (PY-D2): MediaPipe HandLandmarker Tasks
con el mismo `hand_landmarker.task` y los mismos umbrales (contract/features_v2.json).

Reemplaza a Holistic para cualquier extracción nueva. Holistic etiqueta la mano por
la pose; HandLandmarker, por el propio modelo de la mano. Medido sobre los frames de
la webcam, dan el mismo slot para cada mano, pero las coordenadas no son idénticas,
así que entrenar con uno e inferir con el otro es un desajuste evitable.

Devuelve las manos crudas, sin lateralidad corregida ni aspecto: el contrato se
aplica después (`lsa.contrato`), igual que en la app.
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from lsa.contrato import CONTRATO, Mano

# El .task vive en los assets de la app: es la única copia, así no hay dos que se
# desincronicen. Se verifica el hash contra el contrato antes de usarlo.
RUTA_TASK_DEFECTO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
    "TraductorLSA", "app", "src", "main", "assets", "hand_landmarker.task"))


class Extractor:
    def __init__(self, ruta_task: str = RUTA_TASK_DEFECTO):
        from mediapipe.tasks.python import BaseOptions, vision

        with open(ruta_task, "rb") as f:
            modelo = f.read()
        esperado = CONTRATO["extractor"]["sha256"]
        real = hashlib.sha256(modelo).hexdigest()
        if real != esperado:
            raise ValueError(f"{ruta_task} no es el hand_landmarker.task del contrato ({real[:12]} != {esperado[:12]})")
        cfg = CONTRATO["extractor"]
        opciones = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=modelo),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=cfg["num_hands"],
            min_hand_detection_confidence=cfg["min_hand_detection_confidence"],
            min_hand_presence_confidence=cfg["min_hand_presence_confidence"],
            min_tracking_confidence=cfg["min_tracking_confidence"],
        )
        self._detector = vision.HandLandmarker.create_from_options(opciones)

    def manos(self, rgb: np.ndarray) -> list[Mano]:
        """Manos de una imagen RGB uint8 (alto, ancho, 3), sin espejar y ya derecha."""
        import mediapipe as mp

        r = self._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
        salida = []
        for lados, lm in zip(r.handedness, r.hand_landmarks, strict=True):
            mejor = max(lados, key=lambda c: c.score)
            salida.append(Mano(mejor.category_name, float(mejor.score),
                               np.asarray([[p.x, p.y, p.z] for p in lm], np.float32)))
        return salida

    def cerrar(self):
        self._detector.close()
