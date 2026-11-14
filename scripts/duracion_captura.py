"""Duración real de cada toma, por fuente y por dispositivo (auditoría, X4).

La auditoría infería que una toma de la webcam cubre unos 0,5 s y una del teléfono entre
1 y 1,5 s. Acá se mide con lo que guardan los datos:

  - Teléfono (`data/captura_E3/<P>/lsa_samples_v2.jsonl`): la app guarda, por frame, `t_ms`
    = milisegundos desde que empezó la captura (`SystemClock.uptimeMillis()` al analizar el
    frame, menos el instante en que la máquina de estados pasó a CAPTURING), y por toma
    `duracion_ms` (= el `t_ms` del último frame), `fps` y `n_frames`. Son marcas reales.
  - Webcam (`data/landmarks/webcam_v2.jsonl`): `scripts/migrar_v1.py` escribe `t_ms = i·1000/30`,
    `fps = 30` y `duracion_ms = n·1000/30` para todas las tomas. **No son mediciones**: los
    JPG de `frame_actions/` no traen marca de tiempo y `normalize_samples.py` llevó cada toma
    a 15 frames (submuestreando o interpolando) sin anotar cuántos tenía. Lo único medido
    es el instante en que se guardó cada toma (el nombre `sample_<AAMMDDhhmmssffffff>`, que
    `capture_samples.py` arma con `datetime.now()` justo antes de guardar): entre dos tomas
    seguidas de la misma clase pasa la captura entera de la segunda, así que ese intervalo
    es una **cota superior** de su duración.

Solo lee. Escribe `tablas.md` y `resultados.json` en --salida.

Uso, desde la raíz de modelo_LSA:
    python scripts/duracion_captura.py --salida resultados_duracion_captura/2026-11-14
"""
import argparse
import glob
import itertools
import json
import os
import sys
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lsa import dataset_v2

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def medir_toma(m):
    """Lo que dice una toma v2 de su duración. `marcas_reales` es falso para las tomas
    migradas de la webcam, cuyos tiempos se calcularon suponiendo 30 FPS."""
    t = [f["t_ms"] for f in m["frames"]]
    s = m.get("sesion") or {}
    return {"fuente": m.get("fuente", "telefono"), "dispositivo": m["dispositivo"], "persona": s.get("codigo"),
            "bloque": s.get("bloque"), "carpeta": s.get("carpeta"), "label": m["label"],
            "n_frames": len(t), "n_con_mano": sum(1 for f in m["frames"] if f.get("manos")),
            "duracion_ms": m.get("duracion_ms"), "primer_t_ms": t[0] if t else None,
            "entre_primero_y_ultimo_ms": (t[-1] - t[0]) if len(t) > 1 else 0,
            "fps": m.get("fps"), "marcas_reales": m.get("app") != "migrar_v1.py"}


def estadisticas(valores):
    """Mediana, rango y percentiles 10/90 (None si no hay valores)."""
    v = np.asarray([x for x in valores if x is not None], dtype=float)
    if not len(v):
        return None
    return {"n": len(v), "mediana": float(np.median(v)), "min": float(v.min()), "max": float(v.max()),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90))}


def resumir(tomas, claves):
    """Agrupa por `claves` y da las estadísticas de duración, FPS y frames de cada grupo."""
    grupos = {}
    for t in tomas:
        grupos.setdefault(tuple(t[k] for k in claves), []).append(t)
    return [{**dict(zip(claves, k, strict=True)), "tomas": len(g),
             "duracion_ms": estadisticas([t["duracion_ms"] for t in g]),
             "entre_primero_y_ultimo_ms": estadisticas([t["entre_primero_y_ultimo_ms"] for t in g]),
             "fps": estadisticas([t["fps"] for t in g]),
             "n_frames": {str(n): sum(1 for t in g if t["n_frames"] == n) for n in sorted({t["n_frames"] for t in g})},
             "tomas_con_frames_sin_mano": sum(1 for t in g if t["n_con_mano"] < t["n_frames"])}
            for k, g in sorted(grupos.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]


def instante_de_carpeta(nombre):
    """`sample_240917094224729441` -> datetime(2024, 9, 17, 9, 42, 24, 729441)."""
    return datetime.strptime(nombre.removeprefix("sample_"), "%y%m%d%H%M%S%f")


def cotas_por_intervalo(nombres, max_s=60.0):
    """Para tomas de una misma clase: segundos entre cada guardado y el anterior, que acotan
    por arriba la duración de la captura de la toma posterior. Los intervalos de más de
    `max_s` (otra tanda, una pausa) no acotan nada útil y se descartan."""
    instantes = sorted(instante_de_carpeta(n) for n in nombres)
    d = [(b - a).total_seconds() for a, b in itertools.pairwise(instantes)]
    return [x for x in d if x <= max_s]


def _fmt(e, factor=1.0, dec=0):
    if e is None:
        return "—"
    f = lambda x: f"{x * factor:.{dec}f}".replace(".", ",")  # noqa: E731
    return f"{f(e['mediana'])} ({f(e['min'])}-{f(e['max'])}; p10-p90 {f(e['p10'])}-{f(e['p90'])})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salida", required=True)
    args = ap.parse_args()

    rutas = sorted(glob.glob(os.path.join(RAIZ, "data", "captura_E3", "*", "lsa_samples_v2.jsonl")))
    rutas.append(os.path.join(RAIZ, "data", "landmarks", "webcam_v2.jsonl"))
    tomas = [medir_toma(m) for r in rutas for m in dataset_v2.leer(r)
             if (m.get("sesion") or {}).get("codigo") not in (None, "PRUEBA")]

    tel = [t for t in tomas if t["marcas_reales"]]
    web = [t for t in tomas if not t["marcas_reales"]]
    por_dispositivo = resumir(tel, ["dispositivo", "persona"])
    por_bloque = resumir(tel, ["dispositivo", "persona", "bloque"])
    por_frames = resumir(tel, ["dispositivo", "n_frames"])
    webcam = resumir(web, ["dispositivo"])

    # Webcam: cotas superiores por el intervalo entre guardados, por año de grabación.
    cotas = {}
    dir_frames = os.path.join(RAIZ, "frame_actions")
    if os.path.isdir(dir_frames):
        for clase in sorted(os.listdir(dir_frames)):
            nombres = [n for n in os.listdir(os.path.join(dir_frames, clase)) if n.startswith("sample_")]
            por_anio = {}
            for n in nombres:
                por_anio.setdefault(instante_de_carpeta(n).year, []).append(n)
            for anio, ns in por_anio.items():
                cotas.setdefault(anio, []).extend(cotas_por_intervalo(ns))
    cotas_resumen = {str(a): estadisticas(v) for a, v in sorted(cotas.items())}

    r = {"telefono_por_dispositivo": por_dispositivo, "telefono_por_bloque": por_bloque,
         "telefono_por_n_frames": por_frames, "webcam_campos_v2": webcam,
         "webcam_cota_superior_s_por_anio": cotas_resumen}
    os.makedirs(args.salida, exist_ok=True)
    with open(os.path.join(args.salida, "resultados.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2, ensure_ascii=False)

    md = ["# Duración de la captura por toma", "",
          "Generado por `scripts/duracion_captura.py`. Mediana (mín-máx; p10-p90).", "",
          "## Teléfono (marcas de tiempo reales de la app)", "",
          "`duracion_ms`: desde que empieza la captura hasta el último frame. «1.º→último»: entre el primer "
          "y el último frame guardado.", "",
          "| Dispositivo | Persona | Tomas | Frames (tomas) | duracion_ms | 1.º→último ms | FPS |",
          "|---|---|---|---|---|---|---|"]
    for g in por_dispositivo:
        md.append(f"| {g['dispositivo']} | {g['persona']} | {g['tomas']} | {g['n_frames']} | {_fmt(g['duracion_ms'])} "
                  f"| {_fmt(g['entre_primero_y_ultimo_ms'])} | {_fmt(g['fps'], dec=1)} |")
    md += ["", "### Por bloque (A = señas, N = «nada», B = robustez, sesión 2)", "",
           "| Dispositivo | Persona | Bloque | Tomas | Frames (tomas) | duracion_ms | FPS |",
           "|---|---|---|---|---|---|---|"]
    for g in por_bloque:
        md.append(f"| {g['dispositivo']} | {g['persona']} | {g['bloque']} | {g['tomas']} | {g['n_frames']} "
                  f"| {_fmt(g['duracion_ms'])} | {_fmt(g['fps'], dec=1)} |")
    md += ["", "### Por cantidad de frames de la toma", "",
           "| Dispositivo | Frames | Tomas | duracion_ms | FPS |", "|---|---|---|---|---|"]
    for g in por_frames:
        md.append(f"| {g['dispositivo']} | {g['n_frames']} | {g['tomas']} | {_fmt(g['duracion_ms'])} "
                  f"| {_fmt(g['fps'], dec=1)} |")
    md += ["", "## Webcam", ""]
    for g in webcam:
        md.append(f"- {g['dispositivo']}: {g['tomas']} tomas, frames por toma {g['n_frames']}; `duracion_ms` en el "
                  f"archivo: {_fmt(g['duracion_ms'])} y `fps` {_fmt(g['fps'], dec=1)}, **valores supuestos por "
                  "migrar_v1.py (30 FPS), no medidos**. Tomas con algún frame sin mano: "
                  f"{g['tomas_con_frames_sin_mano']}.")
    md += ["", "Cota superior medida (segundos entre el guardado de una toma y el de la anterior de la misma "
           "clase, intervalos ≤ 60 s), por año de grabación:", "",
           "| Año | Intervalos | Mediana s | Mín s | p10 s | p90 s |", "|---|---|---|---|---|---|"]
    for a, e in cotas_resumen.items():
        if e:
            md.append(f"| {a} | {e['n']} | {e['mediana']:.2f} | {e['min']:.2f} | {e['p10']:.2f} | {e['p90']:.2f} |"
                      .replace(".", ","))
    with open(os.path.join(args.salida, "tablas.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
