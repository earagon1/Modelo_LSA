"""Carga todas las fuentes en un solo dataset con el contrato v2 (PY-E1/E3).

Fuentes, todas con el esquema `senar-muestra-v2` (landmarks crudos por frame):
  - webcam:   `data/landmarks/webcam_v2.jsonl` (re-extraída con `scripts/migrar_v1.py`)
  - teléfono: `data/captura_E3/<participante>/lsa_samples_v2.jsonl` (sesión de captura)

Cada muestra trae los metadatos que necesita la validación honesta: persona, fuente,
dispositivo y sesión. Las tomas de PRUEBA se descartan.

Por defecto solo carga los bloques de la sesión 1 (A y N, y la webcam). El bloque B
(robustez, sesión 2) vive en los mismos archivos pero no es entrenamiento: es el test
de `scripts/robustez.py`, que lo pide con `bloques=("B",)`.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

import numpy as np

from lsa import contrato, dataset_v2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEBCAM = os.path.join(RAIZ, "data", "landmarks", "webcam_v2.jsonl")
TELEFONO = os.path.join(RAIZ, "data", "captura_E3", "*", "lsa_samples_v2.jsonl")
WORDS_JSON = os.path.join(RAIZ, "models", "words.json")
NADA = "nada"
# A = tandas de señas, N = "nada", W = webcam (migrar_v1). B (robustez) y C (otro
# dispositivo) son tests y no entran salvo que se pidan: si no, al sumar la sesión 2 a
# los archivos de captura, todo lo que entrena cambiaba sin avisar.
BLOQUES_ENTRENAMIENTO = ("W", "A", "N")


@dataclass
class Dataset:
    X: np.ndarray                    # (N, T, D) float32
    y: np.ndarray                    # (N,) índices en `clases`
    clases: list[str]
    persona: np.ndarray              # P1, P2, ...
    fuente: np.ndarray               # webcam / telefono
    dispositivo: np.ndarray
    sesion: np.ndarray               # grupo para la validación por sesión
    bloque: np.ndarray               # W, A, N, B...
    condicion: np.ndarray            # "luz=...;angulo=...;distancia=..." de la planilla
    orden: np.ndarray                # orden de la toma en la planilla del participante
    excluidas: dict = field(default_factory=dict)

    def subconjunto(self, mascara: np.ndarray) -> Dataset:
        return Dataset(self.X[mascara], self.y[mascara], self.clases, self.persona[mascara],
                       self.fuente[mascara], self.dispositivo[mascara], self.sesion[mascara],
                       self.bloque[mascara], self.condicion[mascara], self.orden[mascara], self.excluidas)

    def resumen(self) -> str:
        from collections import Counter
        partes = [f"{len(self.y)} muestras, {len(self.clases)} clases"]
        for nombre, col in (("persona", self.persona), ("fuente", self.fuente)):
            partes.append(f"{nombre}: {dict(sorted(Counter(col.tolist()).items()))}")
        if self.excluidas:
            partes.append(f"excluidas: {self.excluidas}")
        return " | ".join(partes)


def clases_del_modelo(incluir_nada: bool) -> list[str]:
    """El orden de words.json es el índice de salida del modelo publicado; "nada" va al final."""
    with open(WORDS_JSON, encoding="utf-8") as f:
        clases = list(json.load(f)["word_ids"])
    return [*clases, NADA] if incluir_nada and NADA not in clases else clases


def _grupo_sesion(m: dict) -> str:
    s = m["sesion"]
    if m.get("fuente") == "webcam":
        return f"webcam-{s.get('sesion_id', 'desconocida')}"
    # En el teléfono, cada tanda es una sesión (hubo una pausa entre las dos). Las 10
    # tomas de "nada" se grabaron enseguida de la tanda 2 y van con ella: solas eran
    # un grupo tan chico que StratifiedGroupKFold armaba un pliegue de 10 muestras.
    if s["bloque"] in ("B", "C"):
        return f"{s['codigo']}-{s['bloque']}"
    tanda = 2 if s["bloque"] == "N" else s["tanda"]
    return f"{s['codigo']}-A{tanda}"


def cargar(incluir_nada: bool = True, fuentes: tuple[str, ...] = ("webcam", "telefono"),
           bloques: tuple[str, ...] = BLOQUES_ENTRENAMIENTO) -> Dataset:
    clases = clases_del_modelo(incluir_nada)
    indice = {c: i for i, c in enumerate(clases)}
    rutas = []
    if "webcam" in fuentes and os.path.exists(WEBCAM):
        rutas.append(WEBCAM)
    if "telefono" in fuentes:
        rutas += sorted(glob.glob(TELEFONO))

    X, y, persona, fuente, dispositivo, sesion, bloque, condicion, orden = [], [], [], [], [], [], [], [], []
    excluidas = {"prueba": 0, "fuera_de_sesion": 0, "clase_desconocida": 0, "pocos_frames": 0}
    for ruta in rutas:
        for m in dataset_v2.leer(ruta):
            s = m.get("sesion")
            if s is None:
                excluidas["fuera_de_sesion"] += 1
                continue
            if s["codigo"] == "PRUEBA":
                excluidas["prueba"] += 1
                continue
            if s["bloque"] not in bloques:
                continue
            if m["label"] not in indice:
                excluidas["clase_desconocida"] += 1
                continue
            vectores = np.stack([contrato.frame_a_vector(f, m["img_w"], m["img_h"])
                                 for f in dataset_v2.frames_como_manos(m)])
            seq = contrato.secuencia_desde_vectores_v1(vectores)  # descarta vacíos + remuestrea
            if seq is None:
                excluidas["pocos_frames"] += 1
                continue
            X.append(seq)
            y.append(indice[m["label"]])
            persona.append(s["codigo"])
            fuente.append(m.get("fuente", "telefono"))
            dispositivo.append(m["dispositivo"])
            sesion.append(_grupo_sesion(m))
            bloque.append(s["bloque"])
            condicion.append(s.get("condicion", ""))
            orden.append(int(s.get("orden", 0)))
    return Dataset(np.stack(X).astype(np.float32), np.asarray(y), clases, np.asarray(persona),
                   np.asarray(fuente), np.asarray(dispositivo), np.asarray(sesion), np.asarray(bloque),
                   np.asarray(condicion), np.asarray(orden), {k: v for k, v in excluidas.items() if v})
