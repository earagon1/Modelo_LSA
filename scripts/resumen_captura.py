"""Comparativa de la sesión de captura E3 por participante y dispositivo.

Lee los `lsa_samples_v2.jsonl` de `data/captura_E3/<participante>/` y genera, en
`resultados_captura_E3/`:
  - `resumen_captura.json`: los números;
  - `resumen_captura.md`: las tablas para el informe (y una versión LaTeX booktabs).

Qué compara y por qué importa para la tesis:
  - FPS y frames por toma: muestran la captura adaptativa de la app (10/15/20 frames
    según los FPS, hallazgo H-A02) en dispositivos reales de gama alta y media.
  - % de frames con las dos manos, por seña: dónde le cuesta al detector (las señas
    con las palmas juntas, como «por favor») y cómo cambia según el dispositivo.
  - Duración de la sesión y tomas por persona: el costo real del protocolo.

Solo lee los datos. Uso, desde la raíz de modelo_LSA:
    python scripts/resumen_captura.py
"""
import json
import os
import statistics as st
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa import dataset_v2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTRADA = os.path.join(RAIZ, "data", "captura_E3")
SALIDA = os.path.join(RAIZ, "resultados_captura_E3")


def resumir(muestras):
    sesion = [m for m in muestras if m.get("sesion") and m["sesion"]["codigo"] != "PRUEBA"]
    fps = [m["fps"] for m in sesion]
    frames = Counter(m["n_frames"] for m in sesion)
    dos_manos = defaultdict(list)
    for m in sesion:
        n = [len(f["manos"]) for f in m["frames"]]
        dos_manos[m["label"]].append(sum(k == 2 for k in n) / len(n))
    ts = sorted(m["ts"] for m in sesion)
    return {
        "dispositivos": sorted({m["dispositivo"] for m in sesion}),
        "android": sorted({m["android"] for m in sesion}),
        "tomas": len(sesion),
        "fps_mediana": round(st.median(fps), 1),
        "fps_min": round(min(fps), 1),
        "fps_max": round(max(fps), 1),
        "frames_por_toma": {str(k): v for k, v in sorted(frames.items())},
        "duracion_captura_ms_mediana": st.median(m["duracion_ms"] for m in sesion),
        "minutos_sesion": round((ts[-1] - ts[0]) / 60000, 1),
        "pct_frames_dos_manos": {c: round(100 * sum(v) / len(v)) for c, v in sorted(dos_manos.items())},
    }


def main():
    if not os.path.isdir(ENTRADA):
        print(f"No existe {ENTRADA}")
        return 1
    resultados = {}
    for participante in sorted(os.listdir(ENTRADA)):
        ruta = os.path.join(ENTRADA, participante, "lsa_samples_v2.jsonl")
        if os.path.exists(ruta):
            resultados[participante] = resumir(list(dataset_v2.leer(ruta)))
    if not resultados:
        print("No hay exports v2.")
        return 1

    os.makedirs(SALIDA, exist_ok=True)
    with open(os.path.join(SALIDA, "resumen_captura.json"), "w", encoding="utf-8") as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)

    ps = list(resultados)
    filas = [
        ("Dispositivo", lambda r: ", ".join(r["dispositivos"])),
        ("Android", lambda r: ", ".join(r["android"])),
        ("Tomas", lambda r: str(r["tomas"])),
        ("FPS (mediana)", lambda r: f"{r['fps_mediana']}".replace(".", ",")),
        ("FPS (mín–máx)", lambda r: f"{r['fps_min']}–{r['fps_max']}".replace(".", ",")),
        ("Frames por toma", lambda r: ", ".join(f"{k}: {v}" for k, v in r["frames_por_toma"].items())),
        ("Duración de una captura (ms, mediana)", lambda r: f"{r['duracion_captura_ms_mediana']:.0f}"),
        ("Duración de la sesión (min)", lambda r: f"{r['minutos_sesion']}".replace(".", ",")),
    ]
    md = ["# Sesión de captura E3: comparativa por participante y dispositivo", "",
          "Generado por `scripts/resumen_captura.py` a partir de `data/captura_E3/*/lsa_samples_v2.jsonl`.",
          "No editar a mano.", "",
          "| | " + " | ".join(ps) + " |", "|---|" + "---|" * len(ps)]
    md += [f"| {nombre} | " + " | ".join(fn(resultados[p]) for p in ps) + " |" for nombre, fn in filas]
    clases = sorted({c for r in resultados.values() for c in r["pct_frames_dos_manos"]})
    md += ["", "Lectura: la app captura una cantidad fija de frames según los FPS (10, 15 o 20), así que",
           "la ventana de tiempo que cubre una toma depende del dispositivo. En un equipo rápido 15 frames",
           "pueden durar menos de un segundo, y una seña larga no entra completa; en uno lento la captura",
           "baja a 10 frames. Es el argumento empírico para que el contrato v2 capture por tiempo.",
           "", "## % de frames con las dos manos detectadas, por seña", "",
           "| Seña | " + " | ".join(ps) + " |", "|---|" + "---|" * len(ps)]
    md += [f"| {c} | " + " | ".join(f"{resultados[p]['pct_frames_dos_manos'].get(c, '—')} %" for p in ps) + " |"
           for c in clases]

    tex = [r"\begin{tabular}{l" + "r" * len(ps) + "}", r"\toprule", " & ".join(["", *ps]) + r" \\", r"\midrule"]
    tex += [" & ".join([nombre, *(fn(resultados[p]) for p in ps)]) + r" \\" for nombre, fn in filas]
    tex += [r"\bottomrule", r"\end{tabular}"]
    md += ["", "## Versión LaTeX (booktabs)", "", "```latex", *tex, "```", ""]

    with open(os.path.join(SALIDA, "resumen_captura.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md))
    print("\n".join(md[:len(filas) + 7]))
    print(f"\n-> {SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
