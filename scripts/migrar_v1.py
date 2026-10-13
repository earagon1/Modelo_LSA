"""Re-extrae los frames de la webcam (frame_actions/) con el extractor de la app (PY-D4).

Lee `frame_actions/<clase>/sample_<timestamp>/` (solo lectura) y escribe
`data/landmarks/webcam_v2.jsonl`, con el mismo esquema que el dataset v2 del teléfono
(`senar-muestra-v2`). Así las tres fuentes (webcam, tablet y teléfonos) se cargan y se
convierten al contrato v2 con el mismo código.

- `participant_id`: el de P1, porque la webcam es de Evelin. Así la validación por
  persona no la trata como a alguien distinto de su tablet.
- `sesion.sesion_id`: el día del timestamp de la carpeta (`sample_%y%m%d%H%M%S%f`), que
  es la única marca de sesión que sobrevivió (los .h5 la perdieron).
- Los frames se leen **ordenados por número**. `helpers.get_keypoints` usaba
  `os.listdir` sin ordenar.

Las colas que `normalize_samples.py` borró no se recuperan: cada toma tiene como mucho
sus primeros 15 frames. Se declara en la tesis.

Uso, desde la raíz de modelo_LSA (tarda unos minutos):
    python scripts/migrar_v1.py [--salida data/landmarks/webcam_v2.jsonl]
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa.extractor import Extractor

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTICIPANTE_WEBCAM = "p1_b80dde3cced831aa"  # el seudónimo de la cuenta de Evelin (P1)


def numero(nombre):
    m = re.search(r"(\d+)", nombre)
    return int(m.group(1)) if m else 0


def fecha_de(carpeta):
    try:
        return datetime.strptime(carpeta.removeprefix("sample_")[:12], "%y%m%d%H%M%S")
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=os.path.join(RAIZ, "frame_actions"))
    ap.add_argument("--salida", default=os.path.join(RAIZ, "data", "landmarks", "webcam_v2.jsonl"))
    args = ap.parse_args()
    if os.path.exists(args.salida):
        print(f"{args.salida} ya existe: no se pisa. Borrarlo a mano si se quiere regenerar.")
        return 1
    os.makedirs(os.path.dirname(args.salida), exist_ok=True)

    ext = Extractor()
    total, sin_manos, inicio = 0, 0, time.time()
    with open(args.salida + ".parcial", "w", encoding="utf-8", newline="\n") as out:
        for clase in sorted(os.listdir(args.frames)):
            dir_clase = os.path.join(args.frames, clase)
            for orden, carpeta in enumerate(sorted(os.listdir(dir_clase)), 1):
                dir_toma = os.path.join(dir_clase, carpeta)
                archivos = sorted((f for f in os.listdir(dir_toma) if f.lower().endswith(".jpg")), key=numero)
                frames, ancho, alto = [], 0, 0
                for i, archivo in enumerate(archivos):
                    bgr = cv2.imread(os.path.join(dir_toma, archivo))
                    alto, ancho = bgr.shape[:2]
                    manos = ext.manos(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    frames.append({"t_ms": round(i * 1000 / 30), "manos": [
                        {"e": m.etiqueta, "s": round(m.score, 4),
                         "lm": [round(float(v), 6) for v in m.landmarks.reshape(-1)]} for m in manos]})
                if not any(f["manos"] for f in frames):
                    sin_manos += 1
                fecha = fecha_de(carpeta)
                muestra = {
                    "esquema": "senar-muestra-v2",
                    "uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"webcam/{clase}/{carpeta}")),
                    "ts": int(fecha.timestamp() * 1000) if fecha else 0, "label": clase,
                    "participant_id": PARTICIPANTE_WEBCAM, "fuente": "webcam",
                    "sesion": {"codigo": "P1", "orden": orden, "bloque": "W", "tanda": 0, "condicion": "webcam",
                               "sesion_id": fecha.strftime("%Y-%m-%d") if fecha else "desconocida",
                               "carpeta": carpeta},
                    "dispositivo": "webcam PC", "android": "", "app": "migrar_v1.py",
                    "extractor": "mediapipe-tasks-python-0.10.11", "camara": "webcam",
                    "img_w": ancho, "img_h": alto, "rotacion_deg": 0,
                    "orientacion": "vertical" if alto >= ancho else "horizontal",
                    "fps": 30.0, "duracion_ms": round(len(frames) * 1000 / 30), "n_frames": len(frames),
                    "frames": frames,
                }
                out.write(json.dumps(muestra, separators=(",", ":")) + "\n")
                total += 1
                if total % 100 == 0:
                    print(f"{total} tomas ({time.time() - inicio:.0f} s)", flush=True)
    ext.cerrar()
    os.replace(args.salida + ".parcial", args.salida)
    print(f"{total} tomas -> {args.salida}; {sin_manos} sin ninguna mano detectada; {time.time() - inicio:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
